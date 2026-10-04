#!/usr/bin/env python3
# =============================================================================
# Requirements
# -----------------------------------------------------------------------------
# pip install "transformers>=4.40.0"
# pip install "datasets>=2.18.0"
# pip install "torch>=2.0.0"
# pip install "accelerate>=0.27.0"
# pip install "evaluate>=0.4.1"
# pip install "rouge-score>=0.1.2"
# pip install "scikit-learn>=1.3.0"
# pip install "numpy>=1.24.0"
# pip install "networkx>=3.0"                 # optional: exact PageRank for TextRank
# pip install "sentence-transformers>=2.5.0"  # optional: embedding-based extractive
# pip install "nltk>=3.8"                     # optional: better sentence splitting
# pip install "sentencepiece"                 # required for T5 tokenizers
#
# One-liner:
#   pip install transformers datasets torch accelerate evaluate rouge-score \
#               scikit-learn numpy networkx sentence-transformers nltk sentencepiece
# =============================================================================
"""
Text Summarization System using Transformer Models
==================================================

A complete, modular pipeline that:

  1. Loads and preprocesses the CNN/DailyMail 3.0.0 news corpus.
  2. Runs ABSTRACTIVE summarization with BART (`facebook/bart-large-cnn`)
     or T5 (`t5-small` / `t5-base`), including sentence-aware chunking with
     hierarchical (map-reduce) reduction for documents longer than the
     model's positional limit.
  3. Runs EXTRACTIVE baselines: TF-IDF LexRank, embedding TextRank
     (sentence-transformers), K-Means embedding clustering, and Lead-3.
  4. Scores every system with ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum.
  5. Prints a comparative table plus qualitative side-by-side samples.
  6. Optionally fine-tunes the seq2seq model on CNN/DailyMail.

Usage
-----
    python summarization_system.py                       # default demo (T5-small)
    python summarization_system.py --model facebook/bart-large-cnn --num-samples 50
    python summarization_system.py --do-train --train-samples 2000 --epochs 1
    python summarization_system.py --skip-embeddings     # no sentence-transformers
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import os
import re
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --- Quieter third-party logging ---------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)            #Updated later
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)  #Updated later
logging.getLogger("datasets").setLevel(logging.WARNING)         #Updated later
logging.getLogger("absl").setLevel(logging.WARNING)             #Updated later

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOGGER = logging.getLogger("summarizer")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class PipelineConfig:
    """Central configuration for every stage of the pipeline."""

    # --- Model -------------------------------------------------------------
    model_name: str = "t5-small"           # or facebook/bart-large-cnn, t5-base
    seed: int = 42
    device: Optional[str] = None           # auto-detected when None

    # --- Data --------------------------------------------------------------
    dataset_name: str = "abisee/cnn_dailymail"
    dataset_config: str = "3.0.0"
    text_column: str = "article"
    summary_column: str = "highlights"
    num_eval_samples: int = 20
    num_train_samples: int = 2000
    num_val_samples: int = 200

    # --- Tokenisation / chunking ------------------------------------------
    max_input_tokens: Optional[int] = None  # auto from model config when None
    max_target_tokens: int = 128
    chunk_overlap_sentences: int = 1
    max_reduce_depth: int = 2               # hierarchical summarisation passes

    # --- Generation --------------------------------------------------------
    gen_max_new_tokens: int = 142
    gen_min_new_tokens: int = 40
    num_beams: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    batch_size: int = 4

    # --- Extractive --------------------------------------------------------
    extractive_num_sentences: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    lexrank_damping: float = 0.85
    lexrank_threshold: float = 0.1
    use_embedding_methods: bool = True

    # --- Fine-tuning -------------------------------------------------------
    do_train: bool = False
    output_dir: str = "./checkpoints/summarizer"
    learning_rate: float = 3e-5
    train_batch_size: int = 2
    grad_accum_steps: int = 8
    num_epochs: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: Optional[bool] = None             # auto: True on CUDA

    # --- Misc --------------------------------------------------------------
    verbose_samples: int = 2

    def resolve(self) -> "PipelineConfig":
        """Fill in auto-detected fields. Safe to call more than once."""
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        if self.fp16 is None:
            self.fp16 = self.device == "cuda"
        return self

    @property
    def is_t5(self) -> bool:
        name = self.model_name.lower()
        return "t5" in name and "bart" not in name


# =============================================================================
# 2. TEXT UTILITIES
# =============================================================================
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc",
    "ltd", "co", "corp", "gov", "sen", "rep", "gen", "lt", "col", "capt",
    "u.s", "u.k", "e.g", "i.e", "a.m", "p.m", "no", "fig", "approx",
}

# Only the whitespace is consumed: the lookbehind alternation (fixed-width
# branches, as Python requires) lets a closing quote/bracket sit between the
# terminal punctuation and the space without being swallowed by the split.
_SENT_BOUNDARY_RE = re.compile(
    r"""(?:
          (?<=[.!?])                         # ... end.
        | (?<=[.!?]["'\u201d\u2019)\]])      # ... end."
       )
       \s+                                   # the separator (consumed)
       (?=["'\u201c\u2018(\[]*[A-Z0-9])      # next sentence starts capitalised
    """,
    re.VERBOSE,
)

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_CNN_PREFIX_RE = re.compile(r"^\s*\(?\s*(CNN|Reuters|AP|BBC)\s*\)?\s*(--|—|-)\s*", re.I)
_BYLINE_RE = re.compile(r"^\s*By\s+[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3}\s*(?:\||$)", re.M)


def clean_text(text: str) -> str:
    """Normalise whitespace and strip newswire boilerplate."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = _CNN_PREFIX_RE.sub("", text)
    text = _BYLINE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _nltk_sentences(text: str) -> Optional[List[str]]:
    """Try NLTK's Punkt tokenizer; return None if unavailable."""
    try:
        import nltk  # type: ignore

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        return None


def split_sentences(text: str, min_chars: int = 12) -> List[str]:
    """
    Split `text` into sentences.

    Uses NLTK Punkt when it is installed, otherwise falls back to a regex
    splitter with an abbreviation guard. Never raises; always returns a list.
    """
    text = clean_text(text)
    if not text:
        return []

    sentences = _nltk_sentences(text)
    if sentences is None:
        sentences = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            pieces = _SENT_BOUNDARY_RE.split(line)
            merged: List[str] = []
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                if merged and merged[-1].split():
                    last_token = merged[-1].split()[-1]
                    last_token = last_token.rstrip("\"')]\u201d\u2019").rstrip(".").lower()
                    if last_token in _ABBREVIATIONS:
                        merged[-1] = merged[-1] + " " + piece
                        continue
                merged.append(piece)
            sentences.extend(merged)

    # Glue very short fragments (list bullets, stray initials) onto the
    # preceding sentence so they never become standalone "sentences".
    cleaned: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) < min_chars and cleaned:
            cleaned[-1] = cleaned[-1] + " " + sentence
        else:
            cleaned.append(sentence)
    return cleaned


def to_rouge_lsum_format(summary: str) -> str:
    """ROUGE-Lsum expects one sentence per line."""
    sentences = split_sentences(summary)
    return "\n".join(sentences) if sentences else summary.strip()


def truncate_for_display(text: str, width: int = 100, max_lines: int = 6) -> str:
    wrapped = textwrap.wrap(clean_text(text), width=width)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines] + ["..."]
    return "\n".join("    " + line for line in wrapped)


# =============================================================================
# 3. TABLE RENDERING (no external dependency)
# =============================================================================
def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    align: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> str:
    """Render an ASCII table. `align` entries are 'l', 'r' or 'c'."""
    str_rows = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
    n_cols = len(headers)
    align = list(align) if align else ["l"] + ["r"] * (n_cols - 1)
    align += ["l"] * (n_cols - len(align))

    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row[:n_cols]):
            widths[i] = max(widths[i], len(cell))

    def fmt(cell: str, width: int, mode: str) -> str:
        if mode == "r":
            return cell.rjust(width)
        if mode == "c":
            return cell.center(width)
        return cell.ljust(width)

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines: List[str] = []
    if title:
        total = len(sep)
        lines.append("=" * total)
        lines.append(title.center(total))
        lines.append("=" * total)
    lines.append(sep)
    lines.append("| " + " | ".join(fmt(str(h), widths[i], "c") for i, h in enumerate(headers)) + " |")
    lines.append(sep.replace("-", "="))
    for row in str_rows:
        padded = list(row) + [""] * (n_cols - len(row))
        lines.append("| " + " | ".join(fmt(padded[i], widths[i], align[i]) for i in range(n_cols)) + " |")
    lines.append(sep)
    return "\n".join(lines)


def banner(text: str, char: str = "=", width: int = 88) -> str:
    return f"\n{char * width}\n{text}\n{char * width}"


# =============================================================================
# 4. DATA LOADING & PREPROCESSING
# =============================================================================
class SummarizationDataModule:
    """Loads CNN/DailyMail (or any 2-column summarisation corpus) and tokenises it."""

    T5_PREFIX = "summarize: "

    def __init__(self, config: PipelineConfig, tokenizer: Optional[AutoTokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer
        self._raw = None

    # ------------------------------------------------------------------ load
    def load_raw(self):
        if self._raw is not None:
            return self._raw
        LOGGER.info(
            "Loading dataset '%s' (config=%s) ...",
            self.config.dataset_name,
            self.config.dataset_config,
        )
        try:
            self._raw = load_dataset(self.config.dataset_name, self.config.dataset_config)
        except Exception as exc:  # pragma: no cover - network/permission issues
            raise RuntimeError(
                f"Could not load '{self.config.dataset_name}'. "
                f"Check your internet connection or `datasets` version. Original error: {exc}"
            ) from exc

        available = set(self._raw[list(self._raw.keys())[0]].column_names)
        for col in (self.config.text_column, self.config.summary_column):
            if col not in available:
                raise ValueError(
                    f"Column '{col}' missing from dataset. Available columns: {sorted(available)}"
                )
        LOGGER.info("Splits: %s", {k: len(v) for k, v in self._raw.items()})
        return self._raw

    # ---------------------------------------------------------------- sample
    def get_eval_samples(self, n: Optional[int] = None, split: str = "test") -> List[Dict[str, str]]:
        """Return `n` cleaned (document, reference) pairs."""
        raw = self.load_raw()
        if split not in raw:
            split = list(raw.keys())[-1]
        n = n or self.config.num_eval_samples
        subset = raw[split].select(range(min(n, len(raw[split]))))

        samples: List[Dict[str, str]] = []
        for idx, row in enumerate(subset):
            document = clean_text(row[self.config.text_column])
            reference = clean_text(row[self.config.summary_column])
            if len(document.split()) < 30 or not reference:
                continue
            samples.append({"id": str(idx), "document": document, "reference": reference})
        LOGGER.info("Prepared %d evaluation samples from split '%s'.", len(samples), split)
        return samples

    # ------------------------------------------------------------ tokenising
    def build_tokenized_splits(self):
        """Tokenised train/validation splits ready for `Seq2SeqTrainer`."""
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required to build tokenised splits.")
        raw = self.load_raw()
        cfg = self.config
        prefix = self.T5_PREFIX if cfg.is_t5 else ""
        max_in = cfg.max_input_tokens or 512

        train = raw["train"].select(range(min(cfg.num_train_samples, len(raw["train"]))))
        val_split = "validation" if "validation" in raw else "test"
        val = raw[val_split].select(range(min(cfg.num_val_samples, len(raw[val_split]))))

        def preprocess(batch: Dict[str, List[str]]) -> Dict[str, Any]:
            inputs = [prefix + clean_text(t) for t in batch[cfg.text_column]]
            targets = [clean_text(t) for t in batch[cfg.summary_column]]
            model_inputs = self.tokenizer(
                inputs, max_length=max_in, truncation=True, padding=False
            )
            labels = self.tokenizer(
                text_target=targets,
                max_length=cfg.max_target_tokens,
                truncation=True,
                padding=False,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        remove_cols = train.column_names
        LOGGER.info("Tokenising %d train / %d validation examples ...", len(train), len(val))
        train_tok = train.map(preprocess, batched=True, remove_columns=remove_cols,
                              desc="tokenizing train")
        val_tok = val.map(preprocess, batched=True, remove_columns=remove_cols,
                          desc="tokenizing validation")
        return train_tok, val_tok


# =============================================================================
# 5. ABSTRACTIVE SUMMARIZER (BART / T5) WITH CHUNKING
# =============================================================================
class AbstractiveSummarizer:
    """
    Transformer seq2seq summariser.

    Handles documents longer than the encoder's positional limit by:
      * splitting the document into sentence-aligned token windows (with overlap),
      * summarising every window (the *map* step),
      * concatenating and re-summarising until the result fits in one window
        (the *reduce* step, bounded by `max_reduce_depth`).
    """

    name = "Abstractive (Transformer)"

    def __init__(self, config: PipelineConfig, model=None, tokenizer=None):
        self.config = config.resolve()
        LOGGER.info("Loading model '%s' on device '%s' ...", config.model_name, self.config.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model_name)
        self.model = model or AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        self.model.to(self.config.device)
        self.model.eval()

        self.max_input_tokens = config.max_input_tokens or self._infer_max_input_tokens()
        self.config.max_input_tokens = self.max_input_tokens
        self.prefix = SummarizationDataModule.T5_PREFIX if self.config.is_t5 else ""
        LOGGER.info(
            "Model ready | encoder window = %d tokens | params = %.1fM",
            self.max_input_tokens,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
        )

    # ------------------------------------------------------------- internals
    def _infer_max_input_tokens(self) -> int:
        candidates: List[int] = []
        cfg = self.model.config
        for attr in ("max_position_embeddings", "n_positions"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and 0 < value < 100_000:
                candidates.append(value)
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < 100_000:
            candidates.append(tok_max)
        if not candidates:
            candidates.append(512)
        # Leave headroom for special tokens / task prefix.
        return max(64, min(candidates) - 8)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _hard_split(self, sentence: str, budget: int) -> List[str]:
        """Split a single over-long 'sentence' on raw token boundaries."""
        ids = self.tokenizer(sentence, add_special_tokens=False)["input_ids"]
        pieces = []
        for start in range(0, len(ids), budget):
            pieces.append(
                self.tokenizer.decode(ids[start:start + budget], skip_special_tokens=True).strip()
            )
        return [p for p in pieces if p]

    def chunk_document(self, text: str) -> List[str]:
        """Sentence-aware chunking that respects the encoder token budget."""
        budget = self.max_input_tokens - self._count_tokens(self.prefix) - 4
        budget = max(64, budget)

        if self._count_tokens(text) <= budget:
            return [clean_text(text)]

        sentences = split_sentences(text) or [text]
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            length = self._count_tokens(sentence)
            if length > budget:
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                chunks.extend(self._hard_split(sentence, budget))
                continue
            if current_len + length > budget and current:
                chunks.append(" ".join(current))
                overlap = self.config.chunk_overlap_sentences
                current = current[-overlap:] if overlap > 0 else []
                current_len = sum(self._count_tokens(s) for s in current)
            current.append(sentence)
            current_len += length

        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if c.strip()]

    # ---------------------------------------------------------- generation
    @torch.no_grad()
    def _generate(self, texts: Sequence[str], max_new_tokens: int, min_new_tokens: int) -> List[str]:
        outputs: List[str] = []
        cfg = self.config
        for start in range(0, len(texts), cfg.batch_size):
            batch = [self.prefix + t for t in texts[start:start + cfg.batch_size]]
            encoded = self.tokenizer(
                batch,
                max_length=self.max_input_tokens,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(cfg.device)

            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min(min_new_tokens, max(1, max_new_tokens - 1)),
                num_beams=cfg.num_beams,
                length_penalty=cfg.length_penalty,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                early_stopping=cfg.early_stopping,
            )
            outputs.extend(
                clean_text(s) for s in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return outputs

    def summarize(self, document: str, _depth: int = 0) -> str:
        """Summarise one document, recursing over chunks when necessary."""
        document = clean_text(document)
        if not document:
            return ""

        chunks = self.chunk_document(document)
        if len(chunks) == 1:
            return self._generate(
                chunks, self.config.gen_max_new_tokens, self.config.gen_min_new_tokens
            )[0]

        # MAP: shorter budget per chunk so the concatenation stays manageable.
        per_chunk_max = max(48, self.config.gen_max_new_tokens // max(1, min(len(chunks), 4)))
        per_chunk_min = max(16, per_chunk_max // 3)
        LOGGER.debug("Document split into %d chunks (depth=%d).", len(chunks), _depth)
        partials = self._generate(chunks, per_chunk_max, per_chunk_min)
        combined = " ".join(partials)

        # REDUCE: collapse the partial summaries into a final one.
        if _depth >= self.config.max_reduce_depth:
            return combined
        if self._count_tokens(combined) <= self.max_input_tokens:
            return self._generate(
                [combined], self.config.gen_max_new_tokens, self.config.gen_min_new_tokens
            )[0]
        return self.summarize(combined, _depth=_depth + 1)

    def summarize_batch(self, documents: Sequence[str]) -> List[str]:
        short, short_idx, results = [], [], [""] * len(documents)
        for i, doc in enumerate(documents):
            if len(self.chunk_document(doc)) == 1:
                short.append(clean_text(doc))
                short_idx.append(i)
            else:
                results[i] = self.summarize(doc)          # long docs go through map-reduce
        if short:
            for i, summary in zip(
                short_idx,
                self._generate(short, self.config.gen_max_new_tokens, self.config.gen_min_new_tokens),
            ):
                results[i] = summary
        return results


# =============================================================================
# 6. EXTRACTIVE SUMMARIZERS
# =============================================================================
class BaseExtractiveSummarizer:
    """Common scaffolding: split, score, pick top-k, restore reading order."""

    name = "Extractive"

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _score(self, sentences: List[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def summarize(self, document: str, num_sentences: Optional[int] = None) -> str:
        k = num_sentences or self.config.extractive_num_sentences
        sentences = split_sentences(document)
        if not sentences:
            return clean_text(document)[:600]
        if len(sentences) <= k:
            return " ".join(sentences)
        try:
            scores = np.asarray(self._score(sentences), dtype=float)
        except Exception as exc:
            LOGGER.warning("%s scoring failed (%s); falling back to Lead-%d.", self.name, exc, k)
            return " ".join(sentences[:k])
        if scores.shape[0] != len(sentences) or not np.all(np.isfinite(scores)):
            return " ".join(sentences[:k])
        top = sorted(np.argsort(-scores)[:k])
        return " ".join(sentences[i] for i in top)

    def summarize_batch(self, documents: Sequence[str]) -> List[str]:
        return [self.summarize(doc) for doc in documents]


class LexRankSummarizer(BaseExtractiveSummarizer):
    """Classic LexRank: TF-IDF cosine graph + power-iteration centrality."""

    name = "Extractive (LexRank / TF-IDF)"

    def _score(self, sentences: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
        matrix = vectorizer.fit_transform(sentences)
        similarity = cosine_similarity(matrix)
        np.fill_diagonal(similarity, 0.0)
        similarity[similarity < self.config.lexrank_threshold] = 0.0
        return power_iteration(similarity, damping=self.config.lexrank_damping)


class EmbeddingTextRankSummarizer(BaseExtractiveSummarizer):
    """TextRank over a sentence-transformer similarity graph."""

    name = "Extractive (TextRank / Embeddings)"

    def __init__(self, config: PipelineConfig, encoder=None):
        super().__init__(config)
        self.encoder = encoder or load_sentence_encoder(config)

    def _score(self, sentences: List[str]) -> np.ndarray:
        embeddings = self.encoder.encode(
            sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        similarity = embeddings @ embeddings.T
        similarity = np.clip(similarity, 0.0, None)
        np.fill_diagonal(similarity, 0.0)
        return power_iteration(similarity, damping=self.config.lexrank_damping)


class ClusteringSummarizer(BaseExtractiveSummarizer):
    """K-Means over sentence embeddings; keeps the sentence nearest each centroid."""

    name = "Extractive (KMeans Clustering)"

    def __init__(self, config: PipelineConfig, encoder=None):
        super().__init__(config)
        self.encoder = encoder or load_sentence_encoder(config)

    def summarize(self, document: str, num_sentences: Optional[int] = None) -> str:
        from sklearn.cluster import KMeans

        k = num_sentences or self.config.extractive_num_sentences
        sentences = split_sentences(document)
        if not sentences:
            return clean_text(document)[:600]
        if len(sentences) <= k:
            return " ".join(sentences)

        embeddings = self.encoder.encode(
            sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        kmeans = KMeans(n_clusters=k, random_state=self.config.seed, n_init=10)
        kmeans.fit(embeddings)

        chosen: List[int] = []
        for centroid in kmeans.cluster_centers_:
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            for idx in np.argsort(distances):
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                    break
        return " ".join(sentences[i] for i in sorted(chosen))


class LeadKSummarizer(BaseExtractiveSummarizer):
    """The Lead-K positional baseline — surprisingly strong on news text."""

    name = "Extractive (Lead-3 Baseline)"

    def summarize(self, document: str, num_sentences: Optional[int] = None) -> str:
        k = num_sentences or self.config.extractive_num_sentences
        sentences = split_sentences(document)
        return " ".join(sentences[:k]) if sentences else clean_text(document)[:600]


# ------------------------------------------------------------------ helpers
def power_iteration(
    similarity: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    PageRank-style centrality on a symmetric similarity matrix.

    Uses `networkx.pagerank` when available (exact, well-tested); otherwise
    falls back to a self-contained damped power iteration.
    """
    n = similarity.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    try:
        import networkx as nx  # type: ignore

        graph = nx.from_numpy_array(similarity)
        ranks = nx.pagerank(graph, alpha=damping, max_iter=max_iter, tol=tol, weight="weight")
        return np.array([ranks.get(i, 0.0) for i in range(n)])
    except Exception:
        pass

    row_sums = similarity.sum(axis=1, keepdims=True)
    # Dangling rows become uniform so probability mass is conserved.
    transition = np.where(row_sums > 0, similarity / np.maximum(row_sums, 1e-12), 1.0 / n)
    scores = np.full(n, 1.0 / n)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        updated = teleport + damping * (transition.T @ scores)
        total = updated.sum()
        if total > 0:
            updated /= total
        if np.abs(updated - scores).sum() < tol:
            return updated
        scores = updated
    return scores


_ENCODER_CACHE: Dict[str, Any] = {}


def load_sentence_encoder(config: PipelineConfig):
    """Load (and cache) a sentence-transformers encoder."""
    key = config.embedding_model
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    from sentence_transformers import SentenceTransformer  # type: ignore

    LOGGER.info("Loading sentence encoder '%s' ...", key)
    encoder = SentenceTransformer(key, device=config.resolve().device)
    _ENCODER_CACHE[key] = encoder
    return encoder


def embeddings_available(config: PipelineConfig) -> bool:
    if not config.use_embedding_methods:
        return False
    try:
        load_sentence_encoder(config)
        return True
    except Exception as exc:
        LOGGER.warning("Embedding methods disabled (%s).", exc)
        return False


# =============================================================================
# 7. ROUGE EVALUATION
# =============================================================================
class RougeEvaluator:
    """
    ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum.

    Prefers `evaluate.load("rouge")`; falls back to the `rouge_score` package
    with manual aggregation if `evaluate` cannot reach the hub.
    """

    METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")

    def __init__(self, use_stemmer: bool = True):
        self.use_stemmer = use_stemmer
        self._backend, self._impl = self._init_backend()
        LOGGER.info("ROUGE backend: %s", self._backend)

    def _init_backend(self) -> Tuple[str, Any]:
        try:
            import evaluate  # type: ignore

            return "evaluate", evaluate.load("rouge")
        except Exception as exc:
            LOGGER.warning("`evaluate` unavailable (%s); using `rouge_score` directly.", exc)
        try:
            from rouge_score import rouge_scorer  # type: ignore

            return "rouge_score", rouge_scorer.RougeScorer(
                list(self.METRICS), use_stemmer=self.use_stemmer
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Neither `evaluate` nor `rouge-score` is installed. "
                "Run: pip install evaluate rouge-score"
            ) from exc

    def compute(self, predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
        if not predictions:
            return {m: 0.0 for m in self.METRICS}
        preds = [to_rouge_lsum_format(p) for p in predictions]
        refs = [to_rouge_lsum_format(r) for r in references]

        if self._backend == "evaluate":
            raw = self._impl.compute(
                predictions=preds, references=refs, use_stemmer=self.use_stemmer
            )
            return {m: float(raw.get(m, 0.0)) * 100.0 for m in self.METRICS}

        totals = {m: 0.0 for m in self.METRICS}
        for pred, ref in zip(preds, refs):
            scores = self._impl.score(ref, pred)
            for metric in self.METRICS:
                totals[metric] += scores[metric].fmeasure
        return {m: (totals[m] / len(preds)) * 100.0 for m in self.METRICS}


# =============================================================================
# 8. COMPARISON ORCHESTRATOR
# =============================================================================
@dataclass
class SystemResult:
    name: str
    kind: str                       # "Abstractive" | "Extractive"
    summaries: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    runtime_s: float = 0.0
    avg_words: float = 0.0
    compression: float = 0.0        # summary words / document words


class SummarizationComparator:
    """Runs every registered system over the same samples and scores them."""

    def __init__(self, config: PipelineConfig, evaluator: Optional[RougeEvaluator] = None):
        self.config = config
        self.evaluator = evaluator or RougeEvaluator()
        self.systems: List[Tuple[str, str, Any]] = []

    def register(self, system: Any, kind: str, name: Optional[str] = None) -> None:
        self.systems.append((name or getattr(system, "name", type(system).__name__), kind, system))

    def run(self, samples: Sequence[Dict[str, str]]) -> List[SystemResult]:
        documents = [s["document"] for s in samples]
        references = [s["reference"] for s in samples]
        doc_words = float(np.mean([len(d.split()) for d in documents])) if documents else 1.0

        results: List[SystemResult] = []
        for name, kind, system in self.systems:
            LOGGER.info("Running system: %s (%d documents)", name, len(documents))
            start = time.perf_counter()
            try:
                summaries = list(system.summarize_batch(documents))
            except Exception as exc:
                LOGGER.error("System '%s' failed: %s", name, exc)
                continue
            elapsed = time.perf_counter() - start

            scores = self.evaluator.compute(summaries, references)
            avg_words = float(np.mean([len(s.split()) for s in summaries])) if summaries else 0.0
            results.append(
                SystemResult(
                    name=name,
                    kind=kind,
                    summaries=summaries,
                    scores=scores,
                    runtime_s=elapsed,
                    avg_words=avg_words,
                    compression=(avg_words / doc_words * 100.0) if doc_words else 0.0,
                )
            )
        return results

    # ------------------------------------------------------------- reporting
    @staticmethod
    def report(results: Sequence[SystemResult], samples: Sequence[Dict[str, str]],
               n_examples: int = 2) -> None:
        if not results:
            print("No systems produced results.")
            return

        headers = ["System", "Type", "ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum",
                   "Avg Len", "Compr.%", "Sec/Doc"]
        rows = []
        n_docs = max(1, len(samples))
        for res in results:
            rows.append([
                res.name,
                res.kind,
                f"{res.scores.get('rouge1', 0.0):.2f}",
                f"{res.scores.get('rouge2', 0.0):.2f}",
                f"{res.scores.get('rougeL', 0.0):.2f}",
                f"{res.scores.get('rougeLsum', 0.0):.2f}",
                f"{res.avg_words:.1f}",
                f"{res.compression:.1f}",
                f"{res.runtime_s / n_docs:.2f}",
            ])
        print(banner("COMPARATIVE ROUGE EVALUATION (F1 x 100, higher is better)"))
        print(render_table(headers, rows, align=["l", "l", "r", "r", "r", "r", "r", "r", "r"]))

        # Winner summary ------------------------------------------------------
        best_rows = []
        for metric in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
            best = max(results, key=lambda r: r.scores.get(metric, 0.0))
            best_rows.append([metric.upper(), best.name, f"{best.scores.get(metric, 0.0):.2f}"])
        print("\n" + render_table(["Metric", "Best System", "Score"], best_rows,
                                  align=["l", "l", "r"]))

        abstractive = [r for r in results if r.kind == "Abstractive"]
        extractive = [r for r in results if r.kind == "Extractive"]
        if abstractive and extractive:
            a_best = max(abstractive, key=lambda r: r.scores.get("rougeL", 0.0))
            e_best = max(extractive, key=lambda r: r.scores.get("rougeL", 0.0))
            delta = a_best.scores.get("rougeL", 0.0) - e_best.scores.get("rougeL", 0.0)
            verdict = "abstractive" if delta > 0 else "extractive"
            print(
                f"\nBest abstractive: {a_best.name} (ROUGE-L {a_best.scores['rougeL']:.2f})"
                f"\nBest extractive : {e_best.name} (ROUGE-L {e_best.scores['rougeL']:.2f})"
                f"\nGap             : {delta:+.2f} ROUGE-L in favour of the {verdict} approach."
            )

        # Qualitative examples -------------------------------------------------
        n_examples = min(n_examples, len(samples))
        if n_examples <= 0:
            return
        print(banner("QUALITATIVE SIDE-BY-SIDE COMPARISON"))
        for i in range(n_examples):
            print(f"\n--- Sample #{i + 1} " + "-" * 70)
            print("\n  [SOURCE DOCUMENT]")
            print(truncate_for_display(samples[i]["document"], max_lines=5))
            print("\n  [REFERENCE SUMMARY]")
            print(truncate_for_display(samples[i]["reference"], max_lines=4))
            for res in results:
                if i < len(res.summaries):
                    print(f"\n  [{res.name.upper()}]")
                    print(truncate_for_display(res.summaries[i], max_lines=5))
            print()


# =============================================================================
# 9. FINE-TUNING
# =============================================================================
def _filter_kwargs(cls: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by `cls.__init__` (transformers API drift)."""
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def finetune(config: PipelineConfig, summarizer: AbstractiveSummarizer) -> AbstractiveSummarizer:
    """Fine-tune the seq2seq model on the configured dataset."""
    config = config.resolve()
    set_seed(config.seed)
    tokenizer, model = summarizer.tokenizer, summarizer.model

    data_module = SummarizationDataModule(config, tokenizer=tokenizer)
    train_ds, val_ds = data_module.build_tokenized_splits()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )
    evaluator = RougeEvaluator()

    def compute_metrics(eval_pred) -> Dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = evaluator.compute(decoded_preds, decoded_labels)
        scores["gen_len"] = float(np.mean([len(p.split()) for p in decoded_preds]))
        return {k: round(v, 4) for k, v in scores.items()}

    args_kwargs: Dict[str, Any] = dict(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=max(1, config.train_batch_size),
        gradient_accumulation_steps=config.grad_accum_steps,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        predict_with_generate=True,
        generation_max_length=config.max_target_tokens,
        generation_num_beams=config.num_beams,
        logging_steps=25,
        save_total_limit=1,
        fp16=bool(config.fp16),
        report_to=[],
        seed=config.seed,
    )
    # transformers renamed `evaluation_strategy` -> `eval_strategy` in 4.46.
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    args_kwargs[strategy_key] = "epoch"
    args_kwargs["save_strategy"] = "epoch"
    training_args = Seq2SeqTrainingArguments(**_filter_kwargs(Seq2SeqTrainingArguments, args_kwargs))

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    # `tokenizer=` was deprecated in favour of `processing_class=`.
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**_filter_kwargs(Seq2SeqTrainer, trainer_kwargs))

    print(banner("FINE-TUNING"))
    LOGGER.info("Training on %d examples for %.1f epoch(s) ...", len(train_ds), config.num_epochs)
    trainer.train()

    metrics = trainer.evaluate()
    rows = [[k.replace("eval_", ""), f"{v:.4f}"] for k, v in metrics.items()
            if isinstance(v, (int, float))]
    print("\n" + render_table(["Validation Metric", "Value"], rows, align=["l", "r"],
                              title="POST-TRAINING VALIDATION"))

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    LOGGER.info("Fine-tuned model saved to '%s'.", config.output_dir)

    summarizer.model = trainer.model.to(config.device)
    summarizer.model.eval()
    return summarizer


# =============================================================================
# 10. LONG-DOCUMENT DEMO
# =============================================================================
LONG_DOCUMENT_DEMO = """
The European Space Agency confirmed on Tuesday that its long-delayed Ariane 7 heavy-lift
rocket completed a full static fire test at the Kourou launch complex in French Guiana,
clearing one of the final technical hurdles before an inaugural flight scheduled for the
second quarter of next year. Engineers ran the vehicle's twin Prometheus engines for a
continuous 470 seconds, replicating the thermal and vibrational loads of an operational
ascent. Agency officials described the data as nominal across all 1,200 instrumented
channels. The test had slipped three times since March, twice because of a faulty liquid
oxygen valve and once because of high winds over the coastal pad.

The Ariane 7 programme has been closely watched across the continent because it represents
Europe's attempt to regain independent access to orbit after a gap in launch capability
left several flagship science missions waiting for rides on foreign rockets. Industry
analysts estimate the programme has consumed roughly 4.2 billion euros since its approval,
about nineteen percent above the original budget envelope agreed by member states. Critics
in the European Parliament have questioned whether a partially reusable design chosen in
2021 remains competitive against American vehicles that now fly weekly and recover their
first stages routinely.

Programme director Hélène Marchand rejected that framing during a briefing in Paris. She
argued that the relevant benchmark is sovereign capability rather than raw launch cadence,
noting that eleven of the agency's upcoming Earth observation satellites carry instruments
subject to export restrictions that complicate launching them outside the bloc. Marchand
also pointed to the vehicle's methane-fuelled upper stage, which the agency expects to cut
per-kilogram costs by nearly forty percent relative to the previous generation once the
production line reaches a rate of twelve vehicles per year.

Commercial interest appears to be firming. Three telecommunications operators have signed
launch reservations covering nine missions between the first flight and the end of the
decade, according to contracts disclosed in regulatory filings. A fourth agreement, covering
a constellation of small navigation satellites, remains under negotiation. Suppliers across
Germany, Italy and Spain have begun hiring for serial production, with the prime contractor
reporting that its Bremen facility added roughly 600 manufacturing roles over the past year.

Significant risk remains. The inaugural flight will carry a cluster of university cubesats
rather than a flagship payload, a deliberate choice to limit exposure should the vehicle
fail. Historical data suggests first flights of new heavy-lift rockets fail at a rate
approaching thirty percent. The agency has also not yet demonstrated recovery of the booster
stage, a capability originally promised for the third flight but now described internally as
a goal for the second year of operations. Independent auditors are expected to publish a
review of the programme's cost controls before the end of the year, and several delegations
have signalled that future funding tranches may be tied to its findings.
""".strip()


def run_long_document_demo(summarizer: AbstractiveSummarizer,
                           extractive: BaseExtractiveSummarizer,
                           evaluator: RougeEvaluator) -> None:
    """Show the chunking path on an arbitrary user-supplied document."""
    print(banner("LONG-DOCUMENT DEMO (custom input, chunking path)"))
    document = clean_text(LONG_DOCUMENT_DEMO)
    chunks = summarizer.chunk_document(document)
    n_tokens = summarizer._count_tokens(document)

    info = [
        ["Document words", f"{len(document.split())}"],
        ["Document tokens", f"{n_tokens}"],
        ["Encoder window", f"{summarizer.max_input_tokens}"],
        ["Chunks produced", f"{len(chunks)}"],
        ["Strategy", "single pass" if len(chunks) == 1 else "map-reduce over chunks"],
    ]
    print(render_table(["Property", "Value"], info, align=["l", "r"]))

    abstractive_summary = summarizer.summarize(document)
    extractive_summary = extractive.summarize(document)

    print("\n  [ABSTRACTIVE SUMMARY]")
    print(truncate_for_display(abstractive_summary, max_lines=8))
    print("\n  [EXTRACTIVE SUMMARY]")
    print(truncate_for_display(extractive_summary, max_lines=8))

    # Self-consistency check: how much of the source each summary preserves.
    rows = []
    for label, summary in (("Abstractive", abstractive_summary), ("Extractive", extractive_summary)):
        scores = evaluator.compute([summary], [document])
        rows.append([label, f"{scores['rouge1']:.2f}", f"{scores['rouge2']:.2f}",
                     f"{scores['rougeL']:.2f}", f"{len(summary.split())}"])
    print("\n" + render_table(
        ["System", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Words"], rows,
        align=["l", "r", "r", "r", "r"],
        title="OVERLAP WITH SOURCE DOCUMENT (no reference available)",
    ))


# =============================================================================
# 11. CLI & MAIN
# =============================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer text summarization: abstractive vs extractive comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="t5-small",
                        help="HF model id: t5-small, t5-base, facebook/bart-large-cnn")
    parser.add_argument("--dataset", default="abisee/cnn_dailymail")
    parser.add_argument("--dataset-config", default="3.0.0")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of test documents to evaluate on.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--extractive-sentences", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=None,
                        help="Override the encoder window (default: inferred from the model).")
    parser.add_argument("--max-new-tokens", type=int, default=142)
    parser.add_argument("--min-new-tokens", type=int, default=40)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip sentence-transformers based extractive methods.")
    parser.add_argument("--skip-long-demo", action="store_true")
    parser.add_argument("--examples", type=int, default=2,
                        help="Qualitative samples to print.")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    # Fine-tuning
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--output-dir", default="./checkpoints/summarizer")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        model_name=args.model,
        seed=args.seed,
        device=args.device,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        num_eval_samples=args.num_samples,
        num_train_samples=args.train_samples,
        num_val_samples=args.val_samples,
        max_input_tokens=args.max_input_tokens,
        gen_max_new_tokens=args.max_new_tokens,
        gen_min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        extractive_num_sentences=args.extractive_sentences,
        use_embedding_methods=not args.skip_embeddings,
        do_train=args.do_train,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        train_batch_size=args.train_batch_size,
        num_epochs=args.epochs,
        verbose_samples=args.examples,
    ).resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    set_seed(config.seed)

    print(banner("TEXT SUMMARIZATION SYSTEM — TRANSFORMER vs EXTRACTIVE BASELINES"))
    env_rows = [
        ["Model", config.model_name],
        ["Dataset", f"{config.dataset_name} ({config.dataset_config})"],
        ["Device", config.device],
        ["Torch", torch.__version__],
        ["Eval samples", config.num_eval_samples],
        ["Extractive sentences", config.extractive_num_sentences],
        ["Fine-tuning", "enabled" if config.do_train else "disabled"],
    ]
    print(render_table(["Setting", "Value"], env_rows, align=["l", "l"]))

    # --- 1. Data ------------------------------------------------------------
    data_module = SummarizationDataModule(config)
    samples = data_module.get_eval_samples()
    if not samples:
        LOGGER.error("No usable evaluation samples were produced. Aborting.")
        return 1

    # --- 2. Abstractive model ----------------------------------------------
    abstractive = AbstractiveSummarizer(config)
    data_module.tokenizer = abstractive.tokenizer

    # --- 3. Optional fine-tuning -------------------------------------------
    if config.do_train:
        abstractive = finetune(config, abstractive)

    # --- 4. Extractive baselines -------------------------------------------
    evaluator = RougeEvaluator()
    comparator = SummarizationComparator(config, evaluator)
    comparator.register(
        abstractive, "Abstractive",
        name=f"Abstractive ({config.model_name}{' fine-tuned' if config.do_train else ''})",
    )
    lexrank = LexRankSummarizer(config)
    comparator.register(lexrank, "Extractive")
    comparator.register(LeadKSummarizer(config), "Extractive")

    if embeddings_available(config):
        encoder = load_sentence_encoder(config)
        comparator.register(EmbeddingTextRankSummarizer(config, encoder), "Extractive")
        comparator.register(ClusteringSummarizer(config, encoder), "Extractive")
    else:
        LOGGER.info("Running without embedding-based extractive methods.")

    # --- 5. Run & report ----------------------------------------------------
    print(banner("RUNNING ALL SYSTEMS"))
    results = comparator.run(samples)
    comparator.report(results, samples, n_examples=config.verbose_samples)

    # --- 6. Long-document chunking demo ------------------------------------
    if not args.skip_long_demo:
        run_long_document_demo(abstractive, lexrank, evaluator)

    print(banner("DONE"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        sys.exit(130)

@echo off

call venv\Scripts\activate.bat

python summarization_system.py --num-samples 2 --skip-long-demo

pause
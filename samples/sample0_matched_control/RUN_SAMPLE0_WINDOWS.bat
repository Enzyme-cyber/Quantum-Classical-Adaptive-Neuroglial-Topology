@echo off
cd /d "%~dp0"
python run_sample0.py --seeds 20 --shots 3000 --output-dir sample0_new_run
if errorlevel 1 (
  echo Please read README_SAMPLE0_CN.md for setup and error instructions.
)
pause

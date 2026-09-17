@echo off
if "%QPANDA_QCLOUD_API_KEY%"=="" (
  echo Please set QPANDA_QCLOUD_API_KEY before running.
  pause
  exit /b 1
)
python V7_1_SAMPLE1B_ARITHMETIC_COMPILATION.py --mode qpu --task-config V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_QPU.json --output-dir sample1b_qpu_full
pause

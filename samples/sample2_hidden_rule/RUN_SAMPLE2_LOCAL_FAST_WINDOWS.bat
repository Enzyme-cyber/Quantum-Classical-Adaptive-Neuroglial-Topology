@echo off
cd /d %~dp0
python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode local --reward-suite deterministic --task-config V7_1_SAMPLE2_HIDDEN_RULE_task_LOCAL_FAST.json
pause

@echo off
cd /d %~dp0
python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode surrogate --reward-suite probabilistic --task-config V7_1_SAMPLE2_HIDDEN_RULE_task_PRIMARY.json --n-seeds 20
pause

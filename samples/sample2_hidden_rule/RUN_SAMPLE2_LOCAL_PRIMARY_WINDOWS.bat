@echo off
cd /d %~dp0
python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode local --reward-suite deterministic --task-config V7_1_SAMPLE2_HIDDEN_RULE_task_PRIMARY.json --agents full_spatial_competition,frozen_spatial,no_glia_glia,shuffled_reward_history,equal_affinity,no_competition
pause

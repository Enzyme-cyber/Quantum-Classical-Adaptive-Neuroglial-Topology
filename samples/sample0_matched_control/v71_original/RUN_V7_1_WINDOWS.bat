@echo off
set CORE=V7_1_neuroglial_gate_qpu.py
set PROJECT=V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json
set EVO=V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json

echo [1/3] Resource check
python V7_1_SPATIAL_EVOLVING_NETWORK.py --core %CORE% --project %PROJECT% --evolution-config %EVO% --mode resource
if errorlevel 1 pause & exit /b 1

echo [2/3] Surrogate evolution
python V7_1_SPATIAL_EVOLVING_NETWORK.py --core %CORE% --project %PROJECT% --evolution-config %EVO% --mode surrogate
if errorlevel 1 pause & exit /b 1

echo [3/3] Frozen-plasticity falsification control
python V7_1_SPATIAL_EVOLVING_NETWORK.py --core %CORE% --project %PROJECT% --evolution-config %EVO% --mode surrogate --freeze-plasticity
pause

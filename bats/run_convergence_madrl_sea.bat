@echo off
chcp 65001 >nul

set TOPO=3layers:
set EP=1000:
set INTERVAL=20:

if "%EP%"=="" set EP=1000
if "%INTERVAL%"=="" set INTERVAL=20

set DEVICE=cuda
set EVAL_SEEDS=1
set RL_EVAL_SEEDS=1
set PREFIX=conv_sea_%TOPO%_ep%EP%_int%INTERVAL%_seed1

python full_compare.py ^
  --topo %TOPO% ^
  --ep %EP% ^
  --eval-seeds %EVAL_SEEDS% ^
  --rl-eval-interval %INTERVAL% ^
  --rl-eval-seeds %RL_EVAL_SEEDS% ^
  --skip-milp ^
  --device %DEVICE% ^
  --output-prefix %PREFIX%

pause
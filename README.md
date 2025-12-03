BTC 量化交易模拟与RL对冲系统

项目概述

这是一个完整的比特币（BTC）量化交易模拟系统，包含价格模拟、做市商策略、期权定价和强化学习对冲策略。项目通过6个笔记本逐步构建了一个完整的量化交易生态系统。

项目结构

├─ 01_data_and_stylized_facts.ipynb

├─ 02_btc_simulator_qed_hawkes.ipynb

├─ 03_option_market_layer.ipynb

├─ 04_baseline_mm_strategy.ipynb

├─ 05_rl_hedging_environment.ipynb

├─ 06_rl_training_results.ipynb

├─ 07_rl_vs_baseline_evaluation.ipynb

│

├─ src/

│  ├─ simulator.py        # QED + Hawkes simulator

│  ├─ option_surface.py   # IV surface + BS pricing

│  ├─ mm_strategy.py      # Market-making rule implementation

│  ├─ rl_env.py           # RL MDP environment

│  ├─ rl_agent.py         # RL agent (DQN / actor-critic, etc.)

│  └─ utils.py            # Shared utilities

│

├─ data/

│  ├─ btc_perp_5min.csv   # Raw data (or provided sample)

│  └─ sample_paths.npz    # Saved Monte Carlo paths (for fast evaluation)

│

├─ results/

│  ├─ plots/              # Figures generated from notebooks

│  └─ metrics.json        # Summary metrics (Sharpe, CVaR, tails, etc.)

│

├─ requirements.txt       # Python dependencies

└─ README.md              # Setup & run instructions
```

环境要求
pip install numpy pandas matplotlib scipy torch



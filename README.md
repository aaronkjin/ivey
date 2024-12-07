# Ivey

An AI-powered poker engine with modern reinforcement learning techniques. A CS 229 project.

Agent 1 First-to-Act:
![agent_1_first_action](https://github.com/user-attachments/assets/bd721df4-ee44-4291-b2a9-6633d0f330d8)
Agent 2 vs. Raise:
![agent_2_vs_raise](https://github.com/user-attachments/assets/8c12e9b4-44c8-4dbb-a0f6-3409b673723b)
Agent 2 vs. Call:
<img width="1000" alt="agent_2_vs_call" src="https://github.com/user-attachments/assets/96332662-07f0-4a94-a6d3-25dbe5ef9b12">
Agent 1 vs. Raise:
![agent_1_vs_raise](https://github.com/user-attachments/assets/57d13368-d7e9-47a7-be5b-609178c819f5)

## Getting Started

Prerequisites:

```bash
# create a virtual env
python3 -m venv ivey_env

# activate the virtual env
source ivey_env/bin/activate

# install core dependencies
pip install -r requirements.txt

# run scripts, e.g.
python3 src/poker_push_fold.py

# for charts, run script
python3 src/poker_chart.py
```

To deactivate the virtual environment, simply run:

```bash
deactivate
```

## Background

This study presents a reinforcement learning (RL) approach to developing poker agents, focusing specifically on preflop decision-making in heads-up No-Limit Texas Hold'em. Starting with a simplified push-fold framework, we progressively expanded our poker engine to incorporate more sophisticated betting options while honing in on efficiency and convergence. Our key contribution lies in the development and optimization of a Q-learning based framework that generates effective poker strategies through self-play. We demonstrate that a single-agent learning paradigm, where one agent learns both player positions and periodically synchronizes strategies, achieves better performance compared to traditional dual-agent approaches. Through efficient state space design, precomputed equity calculations, and optimized batch learning, our system learns complex poker concepts such as position play and hand strength evaluation. Our results show that, even with relatively simple RL techniques, poker agents can develop sophisticated preflop strategies that approach theoretical optimality. This work provides insights into both the effectiveness of different training approaches in imperfect information games and the crucial role of state space design in achieving optimal performance.

## Developers

[Aaron Jin](https://github.com/aaronkjin)

[Ryan Cheng](https://github.com/ryachen01)

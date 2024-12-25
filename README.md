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

Ivey presents a RL approach to developing poker agents, focusing on preflop decision-making in heads-up No-Limit Texas Hold'em. Starting with push/fold decisions and expanding to more complex betting, the system uses a single-agent paradigm for Q-learning, where one agent learns both positions (SB + BB). Careful state design and learning optimization led to agents that fundamentally understand poker principles, such as position and hand strength. The results demonstrate that basic RL techniques can discover preflop strategies in-line with GTO when properly implemented.

## Developers

[Aaron Jin](https://github.com/aaronkjin)

[Ryan Cheng](https://github.com/ryachen01)

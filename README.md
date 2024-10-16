# Ivey

An AI-powered poker engine with modern reinforcement learning techniques. A CS 229 project.

## Getting Started

Prerequisites:

```bash
# create a virtual env
python3 -m venv ivey_env

# activate the virtual env
source ivey_env/bin/activate

# install core dependencies
pip install -r requirements.txt

# run script
python3 src/poker_push_fold.py
```

To deactivate the virtual environment, simply run:

```bash
deactivate
```

## Milestone 1

For the first milestone, we are focused on creating a simple bot that can push or fold during the pre-flop action in a simplified Texas Hold'em poker game.
Our initial step was to create a class, "Game," that runs the simulated poker environment. It deals two players (in a heads-up style) their two cards respectively and then takes CHECK, CALL, RAISE, FOLD
as actions. The action is relegated to preflop, where if a player calls a raise we calculate each players equity and assign rewards as such. We currently
are trying to train player 1 to win this game against player 2, where player 1 is our Q-learning agent which acts first, and player 2 is an agent that makes
random moves.

### Motivation

### Method

### Preliminary Experiments

### Next Steps

## Developers

[Aaron Jin](https://github.com/aaronkjin)

[Ryan Cheng](https://github.com/ryachen01)

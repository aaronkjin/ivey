# Ivey

An AI-powered poker engine with modern reinforcement learning techniques. A CS 229 project.

## Getting Started

Prerequisites:

```bash
# install core dependencies
pip install -r requirements.txt
```

## Milestone 1

For the first milestone, we are focused on creating a simple bot that can push or fold during the pre-flop action in a simplified Texas Hold'em poker game.
The first thing we did was created a class "Game" that runs the simulated poker environment. It deals two players cards and then takes CHECK, CALL, RAISE, FOLD
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

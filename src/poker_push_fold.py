from enum import Enum
import json
import random
from treys import Card, Deck, Evaluator

import numpy as np
from collections import defaultdict


class Actions(Enum):
    CHECK = 0
    FOLD = 1
    CALL = 2
    RAISE = 3


class Game:
    def __init__(self):
        self.deck = Deck()
        self.evaluator = Evaluator()

        self.state = []
        self.player1_hand = self.deck.draw(2)
        self.player2_hand = self.deck.draw(2)

        self.winner = None
        self.player1_turn = True

        self.starting_stack = 100.0
        self.pot = 1.5  # SB + BB
        self.cur_raise = 0.5

        self.player1_stack = self.starting_stack - 0.5
        self.player2_stack = self.starting_stack - 1.0

        self.game_over = False
        self.reward = 0

    @staticmethod
    def get_actions(state):
        if len(state) == 0:
            return [Actions.FOLD, Actions.CALL, Actions.RAISE]
        elif state[-1] == Actions.CALL:
            return [Actions.CHECK, Actions.RAISE]
        elif state[-1] == Actions.RAISE:
            return [Actions.FOLD, Actions.CALL]
        else:
            return []

    def make_action(self, action):
<<<<<<< HEAD:poker_push_fold.py
        legal_actions = self.get_actions(self.state)
=======
        legal_actions = self.get_actions()

>>>>>>> 21a2152d81e0c61e9f2baefbd28e76d2811f2647:src/poker_push_fold.py
        if action not in legal_actions:
            raise Exception("Illegal Action")
        
        self.state.append(action)

        if action == Actions.FOLD:
            self.reward = (
                self.player1_stack - self.starting_stack
                if self.player1_turn
                else self.starting_stack - self.player2_stack
            )

            self.game_over = True

        if action == Actions.CHECK:
            self.evaluate_winner()

        if action == Actions.CALL:
            if self.player1_turn:
                self.player1_stack -= self.cur_raise
            else:
                self.player2_stack -= self.cur_raise

            self.pot += self.cur_raise
            self.cur_raise = 0

            if (len(self.state) >= 2) and self.state[-2] == Actions.RAISE:
                self.evaluate_winner()

        if action == Actions.RAISE:
            raise_amount = (
                self.player1_stack if self.player1_turn else self.player2_stack
            )

            self.pot += raise_amount
            self.cur_raise = raise_amount - self.cur_raise

            if self.player1_turn:
                self.player1_stack = 0
            else:
                self.player2_stack = 0

        self.player1_turn = not self.player1_turn

    def random_board(self):
        board = []

        for _ in range(5):
            valid = False

            while not valid:
                rank = random.choice(list("23456789TJQKA"))
                suit = random.choice(list("shdc"))
                card = Card.new(rank + suit)
                valid = True

                if card in board + self.player1_hand + self.player2_hand:
                    valid = False

            board.append(card)

        return board

    def evaluate_winner(self):
        hand1_wins = 0

        for _ in range(100):

            board = self.random_board()

            hand1_score = self.evaluator.evaluate(board, self.player1_hand)
            hand2_score = self.evaluator.evaluate(board, self.player2_hand)

            if hand1_score < hand2_score:
                hand1_wins += 1

        hand1_equity = hand1_wins / 100

        self.reward = self.pot * hand1_equity + self.player1_stack - self.starting_stack
        self.game_over = True

    def get_game_over(self):
        return self.game_over

    def get_reward(self):
        return self.reward

    def get_state(self, player_one):
        state_str = []

        if player_one:
            state_str.append(Card.int_to_pretty_str(self.player1_hand[0]))
            state_str.append(Card.int_to_pretty_str(self.player1_hand[1]))
        else:
            state_str.append(Card.int_to_pretty_str(self.player2_hand[0]))
            state_str.append(Card.int_to_pretty_str(self.player2_hand[1]))

        for action in self.state:
            if action == Actions.CHECK:
                state_str.append("X")
            if action == Actions.FOLD:
                state_str.append("F")
            if action == Actions.CALL:
                state_str.append("C")
            if action == Actions.RAISE:
                state_str.append("R")

        return ",".join(state_str)


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(float)  # Q-values initialized to 0
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = actions  # Available actions
        self.history = []  # To store the (state, action) pairs of each game

    def choose_action(self, state, legal_actions):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(legal_actions)
        else:
            q_values = [self.q_table[(state, action)] for action in legal_actions]
            max_q = max(q_values)
            max_q_actions = [
                action
                for action in legal_actions
                if self.q_table[(state, action)] == max_q
            ]

            return random.choice(max_q_actions)

<<<<<<< HEAD:poker_push_fold.py
    def update(self, final_reward):
        # Iterate over the game history to update Q-values for all actions taken
        for t in reversed(range(len(self.history))):
            state, action = self.history[t]
            if t == len(self.history) - 1:
                # The final step: reward is the final outcome of the game
                reward = final_reward
            else:
                # Intermediate steps: reward is 0 and the value comes from future states
                next_state, _ = self.history[t + 1]
                reward = 0

                future_q = max(
                    [
                        self.q_table[entry]
                        for entry in self.q_table
                        if entry[0] == next_state
                    ]
                )
                reward += self.gamma * future_q

            # Q-learning update rule
            current_q = self.q_table[(state, action)]
            self.q_table[(state, action)] = current_q + self.alpha * (
                reward - current_q
            )

        # Clear the history after each game
        self.history = []
=======
    def update(self, state, action, reward, next_state, next_legal_actions):
        # Find max Q-value for the next state
        max_next_q = max(
            [
                self.q_table[(next_state, next_action)]
                for next_action in next_legal_actions
            ],
            default=0,
        )

        # Q-learning update rule
        current_q = self.q_table[(state, action)]
        self.q_table[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
>>>>>>> 21a2152d81e0c61e9f2baefbd28e76d2811f2647:src/poker_push_fold.py


# Main game loop where player 1 learns using Q-learning and player 2 plays randomly
def play_game(agent):
    game = Game()

    while game.get_game_over() == False:
        if game.player1_turn:
            state = game.get_state(player_one=True)
            legal_actions = Game.get_actions(game.state)
            action = agent.choose_action(state, legal_actions)
            game.make_action(action)
            agent.history.append((state, action))

        else:
            legal_actions = Game.get_actions(game.state)
            action = random.choice(legal_actions)
            game.make_action(action)

    reward = game.get_reward()
    agent.update(final_reward=reward)

    return game.get_reward()


def simulate_game(agent):
    game = Game()

    while game.get_game_over() == False:
        state = game.get_state(player_one=True)
<<<<<<< HEAD:poker_push_fold.py
        legal_actions = Game.get_actions(game.state)
=======
        legal_actions = game.get_actions()
        
>>>>>>> 21a2152d81e0c61e9f2baefbd28e76d2811f2647:src/poker_push_fold.py
        if game.player1_turn:
            action = agent.choose_action(state, legal_actions)
            game.make_action(action)
        else:
            action = random.choice(legal_actions)
            game.make_action(action)

        print(str(action))

    Card.print_pretty_cards(game.player1_hand)
    Card.print_pretty_cards(game.player2_hand)
    print(game.get_reward())


agent = QLearningAgent(
    actions=[Actions.CHECK, Actions.FOLD, Actions.CALL, Actions.RAISE]
)


total_reward = 0

for episode in range(100000):
    reward = play_game(agent)
    total_reward += reward

    if episode % 100 == 0:
        print(total_reward)


print(f"Total BB Won: {total_reward}")

for state in agent.q_table.keys():
    for action in state:
        print(action)
    print(agent.q_table[state])
    print("")
# print(agent.q_table)

# with open("result.json", "w") as fp:
#     json.dump(agent.q_table, fp)

# for _ in range(1000):
# simulate_game(agent)

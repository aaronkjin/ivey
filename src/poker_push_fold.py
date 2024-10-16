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

    def get_actions(self):
        if len(self.state) == 0:
            return [Actions.FOLD, Actions.CALL, Actions.RAISE]
        elif self.state[-1] == Actions.CALL:
            return [Actions.CHECK, Actions.RAISE]
        elif self.state[-1] == Actions.RAISE:
            return [Actions.FOLD, Actions.CALL]
        else:
            return []

    def make_action(self, action):
        legal_actions = self.get_actions()

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
        self.q_table = defaultdict(float)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = actions

    def choose_action(self, state, legal_actions):
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


# Main game loop where player 1 learns using Q-learning and player 2 plays randomly
def play_game(agent):
    game = Game()

    while game.get_game_over() == False:
        if game.player1_turn:
            state = game.get_state(player_one=True)
            legal_actions = game.get_actions()
            action = agent.choose_action(state, legal_actions)
            game.make_action(action)

            if game.get_game_over():
                reward = game.get_reward()
                next_state = None
                next_legal_actions = []
            else:
                reward = 0
                next_state = game.get_state(player_one=True)
                next_legal_actions = game.get_actions()

            agent.update(state, action, reward, next_state, next_legal_actions)

        else:
            legal_actions = game.get_actions()
            action = random.choice(legal_actions)
            game.make_action(action)

            if game.get_game_over():
                reward = game.get_reward()
                next_state = None
                next_legal_actions = []
                agent.update(state, action, reward, next_state, next_legal_actions)

    return game.get_reward()


def simulate_game(agent):
    game = Game()

    while game.get_game_over() == False:
        state = game.get_state(player_one=True)
        legal_actions = game.get_actions()
        
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

for episode in range(1000000):
    reward = play_game(agent)
    total_reward += reward

    if episode % 100 == 0:
        print(total_reward)

print(f"Total BB Won: {total_reward}")

with open("result.json", "w") as fp:
    json.dump(agent.q_table, fp)

for _ in range(1000):
    simulate_game(agent)

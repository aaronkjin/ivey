from enum import Enum
import json
import random
from treys import Card, Deck, Evaluator

import numpy as np
from collections import defaultdict
from tqdm import tqdm

evaluator = Evaluator()


class Actions(Enum):
    CHECK = 0
    FOLD = 1
    CALL = 2
    RAISE = 3


class Game:
    def __init__(self):
        self.deck = Deck()

        self.state = []
        self.player1_hand = self.deck.draw(2)
        self.player2_hand = self.deck.draw(2)

        self.remaining_cards = self.deck.cards.copy()

        self.winner = None
        self.player1_turn = True

        self.starting_stack = 20.0
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
        legal_actions = self.get_actions(self.state)
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
        for _ in range(25):
            board = random.sample(self.remaining_cards, 5)

            hand1_score = evaluator.evaluate(board, self.player1_hand)
            hand2_score = evaluator.evaluate(board, self.player2_hand)
            if hand1_score < hand2_score:
                hand1_wins += 1

        hand1_equity = hand1_wins / 25

        self.reward = self.pot * hand1_equity + self.player1_stack - self.starting_stack
        self.game_over = True

    def get_game_over(self):
        return self.game_over

    def get_reward(self):
        return self.reward

    def hand_to_string(self, hand):
        rank = list("23456789TJQKA")
        rank_int_0 = Card.get_rank_int(hand[0])
        suit_int_0 = Card.get_suit_int(hand[0])
        rank_int_1 = Card.get_rank_int(hand[1])
        suit_int_1 = Card.get_suit_int(hand[1])

        hand_str = ""
        if rank_int_0 > rank_int_1:
            hand_str = rank[rank_int_0] + rank[rank_int_1]
        else:
            hand_str = rank[rank_int_1] + rank[rank_int_0]
        hand_str += "S" if suit_int_0 == suit_int_1 else "O"
        return hand_str

    def get_state(self, player_one):
        state_str = []
        if player_one:
            state_str.append(self.hand_to_string(self.player1_hand))
        else:
            state_str.append(self.hand_to_string(self.player2_hand))

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

    def update(self, final_reward):
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
                    self.q_table[entry]
                    for entry in self.q_table
                    if entry[0] == next_state
                )
                reward += self.gamma * future_q

            # Q-learning update rule
            current_q = self.q_table[(state, action)]
            if action == Actions.FOLD:
                self.q_table[(state, action)] = reward
            else:
                self.q_table[(state, action)] = current_q + self.alpha * (
                    reward - current_q
                )

        # Clear the history after each game
        self.history = []

    def print_table(self, save_file=None):
        for state_action in self.q_table.keys():
            state, action = state_action
            print(f"State: {state}", file=save_file)
            print(f"Action: {action}", file=save_file)
            print(f"Q-value: {self.q_table[state_action]}", file=save_file)
            print("", file=save_file)

    def load_table(self, filename):

        with open(filename, "r", encoding="utf-8") as f:
            state = None
            action_str = None
            for line in f:
                line = line.strip()
                if line.startswith("State: "):
                    state = line[len("State: ") :]
                elif line.startswith("Action: "):
                    action_str = line[len("Action: ") :]
                elif line.startswith("Q-value: "):
                    q_value_str = line[len("Q-value: ") :]
                    q_value = float(q_value_str)
                    if action_str == "Actions.CALL":
                        action = Actions.CALL
                    if action_str == "Actions.CHECK":
                        action = Actions.CHECK
                    if action_str == "Actions.FOLD":
                        action = Actions.FOLD
                    if action_str == "Actions.RAISE":
                        action = Actions.RAISE

                    self.q_table[(state, action)] = q_value

        print(f"Q-table successfully loaded from '{filename}'.")


# Main game loop where both players learn using Q-learning
def play_game(agent1, agent2):
    game = Game()

    while not game.get_game_over():
        if game.player1_turn:
            state = game.get_state(player_one=True)
            legal_actions = Game.get_actions(game.state)
            action = agent1.choose_action(state, legal_actions)
            game.make_action(action)
            agent1.history.append((state, action))
        else:
            state = game.get_state(player_one=False)
            legal_actions = Game.get_actions(game.state)
            action = agent2.choose_action(state, legal_actions)
            game.make_action(action)
            agent2.history.append((state, action))

    reward = game.get_reward()
    # if game.hand_to_string(game.player1_hand) == "AAO":
    #     print(reward, agent1.history)
    agent1.update(final_reward=reward)
    agent2.update(final_reward=-reward)

    return reward


def simulate_game(agent1, agent2):
    game = Game()

    while not game.get_game_over():
        if game.player1_turn:
            state = game.get_state(player_one=True)
            legal_actions = Game.get_actions(game.state)
            action = agent1.choose_action(state, legal_actions)
            game.make_action(action)
        else:
            state = game.get_state(player_one=False)
            legal_actions = Game.get_actions(game.state)
            action = agent2.choose_action(state, legal_actions)
            game.make_action(action)

        print(str(action))

    Card.print_pretty_cards(game.player1_hand)
    Card.print_pretty_cards(game.player2_hand)
    print(game.get_reward())


def main():
    agent1 = QLearningAgent(
        actions=[Actions.CHECK, Actions.FOLD, Actions.CALL, Actions.RAISE], epsilon=0.0
    )
    agent2 = QLearningAgent(
        actions=[Actions.CHECK, Actions.FOLD, Actions.CALL, Actions.RAISE], epsilon=0.0
    )

    for epoch in range(100):
        epoch_reward = 0
        for _ in tqdm(range(10000)):
            reward = play_game(agent1, agent2)
            epoch_reward += reward
        print(
            f"Epoch {epoch}, Total Reward: {epoch_reward}, BB per hand: {epoch_reward / 10000}"
        )

    with open("q_table_agent1.txt", "w", encoding="utf-8") as file:
        agent1.print_table(save_file=file)
    with open("q_table_agent2.txt", "w", encoding="utf-8") as file:
        agent2.print_table(save_file=file)

    for i in range(10):
        simulate_game(agent1, agent2)


if __name__ == "__main__":
    main()

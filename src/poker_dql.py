from enum import Enum
import copy
import itertools
import random
from treys import Card, Deck, Evaluator
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
            if (len(state) >= 2) and state[-2] == Actions.RAISE:
                return []
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

    @staticmethod
    def get_state_str_from_array(state):
        state_str = []
        for action in state:
            if action == Actions.CHECK:
                state_str.append("X")
            if action == Actions.FOLD:
                state_str.append("F")
            if action == Actions.CALL:
                state_str.append("C")
            if action == Actions.RAISE:
                state_str.append("R")

        return ",".join(state_str)

    @staticmethod
    def state_str_to_state_array(state_str):
        parts = state_str.split(",")
        # The first part is the hand string
        hand_str = parts[0]
        # The rest are action letters
        action_letters = parts[1:]
        letter_to_action = {
            "X": Actions.CHECK,
            "F": Actions.FOLD,
            "C": Actions.CALL,
            "R": Actions.RAISE,
        }
        state_array = [letter_to_action[letter] for letter in action_letters]
        return state_array


class DQNAgent:
    def __init__(
        self,
        action_space,
        alpha=0.005,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=256,
        memory_size=10000,
        target_update_freq=1024,
    ):
        self.action_space = action_space  # List of Actions
        self.action_size = len(action_space)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = []  # Experience replay buffer
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        self.history = []

        # State size is 43 as per our state representation
        self.state_size = 27 + 16  # 43

        # Define the main network and the target network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        self.buffer_cnt = 0

    def _build_model(self):
        # Build the neural network model
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )
        return model

    def update_target_network(self):
        # Update the target network weights
        self.target_model.load_state_dict(self.model.state_dict())

    def load_experiences(self, final_reward):
        for t in reversed(range(len(self.history))):
            state, action = self.history[t]
            if t == len(self.history) - 1:
                # The final step: reward is the final outcome of the game
                reward = final_reward
                next_state = None
            else:
                # Intermediate steps: reward is 0 and the value comes from future states
                next_state, _ = self.history[t + 1]
                reward = 0
            self.remember(state, action, reward, next_state, (next_state is None))
        self.history = []

    def remember(self, state_str, action, reward, next_state_str, done):
        # Convert states to tensors
        state_vec = self.state_to_tensor(state_str)
        next_state_vec = (
            self.state_to_tensor(next_state_str)
            if next_state_str is not None
            else np.zeros(self.state_size)
        )
        # Add experience to replay buffer
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state_vec, action.value, reward, next_state_vec, done))

    def choose_action(self, state_str, legal_actions):
        state_vec = self.state_to_tensor(state_str)
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(legal_actions)
        else:
            # State is a tensor
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(
                0
            )  # Add batch dimension
            q_values = self.model(state_tensor).detach().numpy()[0]
            # Mask illegal actions
            masked_q_values = np.full(self.action_size, -np.inf)
            for action in legal_actions:
                masked_q_values[action.value] = q_values[action.value]
            max_q = np.max(masked_q_values)
            max_actions = [
                action
                for action in legal_actions
                if masked_q_values[action.value] == max_q
            ]
            return random.choice(max_actions)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample minibatch
        minibatch = random.sample(self.memory, self.batch_size)

        reward_sums = defaultdict(float)  # Sum of rewards for each (state, action) pair
        reward_counts = defaultdict(
            int
        )  # Count occurrences for each (state, action) pair
        for state_vec, action_idx, reward, next_state_vec, done in minibatch:
            state_vec_str = self.tensor_to_state(state_vec)
            reward_sums[(state_vec_str, action_idx)] += reward
            reward_counts[(state_vec_str, action_idx)] += 1

            if not done:
                next_state_tensor = torch.FloatTensor(next_state_vec)
                next_state_arr = Game.state_str_to_state_array(
                    self.tensor_to_state(next_state_vec)
                )
                legal_actions = Game.get_actions(next_state_arr)
                legal_actions_idx = [action.value for action in legal_actions]

                # Use target network to get next Q-values
                next_q_values = (
                    self.target_model(next_state_tensor)
                    .detach()
                    .numpy()[legal_actions_idx]
                )
                reward_sums[(state_vec_str, action_idx)] += max(next_q_values)
            # else:
            #     if "AA" in state_vec_str:
            #         print(
            #             reward_sums[(state_vec_str, action_idx)],
            #             reward_counts[(state_vec_str, action_idx)],
            #         )

        # Prepare batches
        state_batch = []
        target_q_batch = []

        for state_vec_str, action_idx in reward_counts.keys():
            state_vec = self.state_to_tensor(state_vec_str)
            state_tensor = torch.FloatTensor(state_vec)
            target = self.model(state_tensor).detach().numpy()
            target[action_idx] = (
                reward_sums[(state_vec_str, action_idx)]
                / reward_counts[(state_vec_str, action_idx)]
            )
            # if "AA" in state_vec_str:
            #     print(state_vec_str, action_idx, target[action_idx])
            state_batch.append(state_vec)
            target_q_batch.append(target)

        # for state_vec, action_idx, reward, next_state_vec, done in minibatch:
        #     state_tensor = torch.FloatTensor(state_vec)
        #     target = self.model(state_tensor).detach().numpy()
        #     if done:
        #         target[action_idx] = reward
        #         # print(
        #         #     self.tensor_to_state(state_vec),
        #         #     action_idx,
        #         #     reward,
        #         # )
        #     else:
        #         next_state_tensor = torch.FloatTensor(next_state_vec)
        #         next_state_arr = Game.state_str_to_state_array(
        #             self.tensor_to_state(next_state_vec)
        #         )
        #         legal_actions = Game.get_actions(next_state_arr)
        #         legal_actions_idx = [action.value for action in legal_actions]

        #         # Use target network to get next Q-values
        #         next_q_values = (
        #             self.target_model(next_state_tensor)
        #             .detach()
        #             .numpy()[legal_actions_idx]
        #         )
        #         target[action_idx] = reward + self.gamma * np.max(next_q_values)
        #         # print(
        #         #     self.tensor_to_state(state_vec),
        #         #     self.tensor_to_state(next_state_vec),
        #         #     action_idx,
        #         #     target[action_idx],
        #         #     next_q_values,
        #         # )

        #     state_batch.append(state_vec)
        #     target_q_batch.append(target)

        # Convert batches to tensors
        state_batch_tensor = torch.FloatTensor(np.array(state_batch))
        target_q_batch_tensor = torch.FloatTensor(np.array(target_q_batch))

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = self.model(state_batch_tensor)
        loss = self.loss_fn(outputs, target_q_batch_tensor)
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps_done += 128
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

    def state_to_tensor(self, state_str):
        # state_str example: 'AKS,C,R'
        rank_to_index = {
            "2": 0,
            "3": 1,
            "4": 2,
            "5": 3,
            "6": 4,
            "7": 5,
            "8": 6,
            "9": 7,
            "T": 8,
            "J": 9,
            "Q": 10,
            "K": 11,
            "A": 12,
        }
        action_to_index = {"X": 0, "F": 1, "C": 2, "R": 3}
        # Split state string
        parts = state_str.split(",")
        hand_str = parts[0]
        action_history_list = parts[1:] if len(parts) > 1 else []
        # Hand vector
        card1_rank = hand_str[0]
        card2_rank = hand_str[1]
        suited = 1 if hand_str[2] == "S" else 0
        card1_vec = np.zeros(13)
        card1_vec[rank_to_index[card1_rank]] = 1
        card2_vec = np.zeros(13)
        card2_vec[rank_to_index[card2_rank]] = 1
        suited_vec = np.array([suited])
        hand_vec = np.concatenate([card1_vec, card2_vec, suited_vec])  # shape (27,)

        # Action history vector
        max_history_length = 4
        action_vec = np.zeros(max_history_length * 4)
        for i, action in enumerate(action_history_list):
            if i >= max_history_length:
                break
            idx = i * 4 + action_to_index[action]
            action_vec[idx] = 1
        # Total state vector
        state_vec = np.concatenate([hand_vec, action_vec])  # shape (43,)
        return state_vec

    def tensor_to_state(self, state_tensor):
        # Assuming state_tensor is a numpy array of shape (43,)
        rank_indices = {
            0: "2",
            1: "3",
            2: "4",
            3: "5",
            4: "6",
            5: "7",
            6: "8",
            7: "9",
            8: "T",
            9: "J",
            10: "Q",
            11: "K",
            12: "A",
        }
        index_to_action = {0: "X", 1: "F", 2: "C", 3: "R"}

        # Hand representation
        card1_vec = state_tensor[0:13]
        card2_vec = state_tensor[13:26]
        suited_vec = state_tensor[26]

        # Get indices where value is maximum
        card1_rank_idx = np.argmax(card1_vec)
        card2_rank_idx = np.argmax(card2_vec)
        suited = int(round(suited_vec))

        card1_rank = rank_indices[card1_rank_idx]
        card2_rank = rank_indices[card2_rank_idx]
        suited_str = "S" if suited == 1 else "O"

        # Order the ranks correctly (higher rank first)
        rank_order = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        card1_rank_value = rank_order.index(card1_rank)
        card2_rank_value = rank_order.index(card2_rank)

        if card1_rank_value >= card2_rank_value:
            hand_str = card1_rank + card2_rank + suited_str
        else:
            hand_str = card2_rank + card1_rank + suited_str

        # Action history
        action_vec = state_tensor[27:]
        action_history_list = []
        for i in range(0, 16, 4):
            action_slice = action_vec[i : i + 4]
            if np.sum(action_slice) == 0:
                break  # No more actions
            action_idx = np.argmax(action_slice)
            action_str = index_to_action[action_idx]
            action_history_list.append(action_str)

        # Build state string
        state_parts = [hand_str] + action_history_list
        state_str = ",".join(state_parts)
        return state_str

    def get_q_values(self, state_str):
        state_vec = self.state_to_tensor(state_str)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)  # Add batch dimension
        q_values = self.model(state_tensor).detach().numpy()[0]
        return q_values

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_network()


def play_game(agent1, agent2):
    game = Game()

    while not game.get_game_over():
        if game.player1_turn:
            state_str = game.get_state(player_one=True)
            legal_actions = Game.get_actions(game.state)
            action = agent1.choose_action(state_str, legal_actions)
            game.make_action(action)
            agent1.history.append((state_str, action))
            # next_state_str = (
            #     game.get_state(player_one=True) if not game.get_game_over() else None
            # )
            # reward = game.get_reward() if game.get_game_over() else 0
            # done = game.get_game_over()
            # agent1.remember(state_str, action, reward, next_state_str, done)
            # if done:
            #     agent1.replay()
        else:
            state_str = game.get_state(player_one=False)
            legal_actions = Game.get_actions(game.state)
            action = agent2.choose_action(state_str, legal_actions)
            game.make_action(action)
            # next_state_str = (
            #     game.get_state(player_one=False) if not game.get_game_over() else None
            # )
            # reward = -game.get_reward() if game.get_game_over() else 0
            # done = game.get_game_over()
            # agent2.remember(state_str, action, reward, next_state_str, done)
            # if done:
            #     agent2.replay()
            agent2.history.append((state_str, action))

    reward = game.get_reward()
    agent1.load_experiences(final_reward=reward)
    agent2.load_experiences(final_reward=-reward)

    agent1.buffer_cnt += 1
    agent2.buffer_cnt += 1

    if agent1.buffer_cnt >= 128:
        agent1.replay()
        agent1.buffer_cnt = 0
    if agent2.buffer_cnt >= 128:
        agent2.replay()
        agent2.buffer_cnt = 0
    return reward


def simulate_game(agent1, agent2):
    # Set epsilon to zero for evaluation
    original_epsilon1 = agent1.epsilon
    original_epsilon2 = agent2.epsilon
    agent1.epsilon = 0
    agent2.epsilon = 0

    game = Game()

    while not game.get_game_over():
        if game.player1_turn:
            state_str = game.get_state(player_one=True)
            legal_actions = Game.get_actions(game.state)
            action = agent1.choose_action(state_str, legal_actions)
            game.make_action(action)
        else:
            state_str = game.get_state(player_one=False)
            legal_actions = Game.get_actions(game.state)
            action = agent2.choose_action(state_str, legal_actions)
            game.make_action(action)

    # Restore original epsilon
    agent1.epsilon = original_epsilon1
    agent2.epsilon = original_epsilon2

    return game.get_reward()


def generate_starting_hands():
    ranks = list("23456789TJQKA")
    starting_hands = []

    # Pairs
    for rank in ranks:
        hand_str = rank + rank + "O"  # Pairs are offsuit
        starting_hands.append(hand_str)

    # Non-pair combinations
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            rank1 = ranks[j]  # Higher rank
            rank2 = ranks[i]  # Lower rank
            hand_str_suited = rank1 + rank2 + "S"
            hand_str_offsuit = rank1 + rank2 + "O"
            starting_hands.append(hand_str_suited)
            starting_hands.append(hand_str_offsuit)

    return starting_hands


def get_all_possible_states():
    starting_hands = generate_starting_hands()
    all_states = set()

    stack = [[]]
    while len(stack) != 0:
        cur = stack.pop()
        cur_str = Game.get_state_str_from_array(cur)
        if cur_str not in all_states:
            legal_actions = Game.get_actions(cur)
            if len(legal_actions) > 0:
                all_states.add(cur_str)
            for action in legal_actions:
                cur_copy = copy.deepcopy(cur)
                cur_copy.append(action)
                stack.append(cur_copy)

    player1_states = [
        state for state in all_states if len(state.replace(",", "")) % 2 == 0
    ]
    player2_states = [
        state for state in all_states if len(state.replace(",", "")) % 2 == 1
    ]
    return (
        [
            (f"{x[0]},{x[1]}" if x[1] != "" else f"{x[0]}")
            for x in list(itertools.product(starting_hands, player1_states))
        ],
        [
            (f"{x[0]},{x[1]}" if x[1] != "" else f"{x[0]}")
            for x in list(itertools.product(starting_hands, player2_states))
        ],
    )


def calculate_q_values_for_starting_hands(agent, save_file, player1):
    all_states1, all_states2 = get_all_possible_states()
    all_states = all_states1 if player1 else all_states2
    legal_actions = [Actions.CHECK, Actions.FOLD, Actions.CALL, Actions.RAISE]

    # with open(filename, "w") as f:
    #     f.write("Hand,CHECK, FOLD,CALL,RAISE\n")
    q_table = defaultdict(float)
    for hand_str in all_states:
        state_str = hand_str  # No action history
        q_values = agent.get_q_values(state_str)
        legal_actions = Game.get_actions(Game.state_str_to_state_array(state_str))

        # q_values_dict = {}
        for action in legal_actions:
            q_value = q_values[action.value]
            q_table[(state_str, action.name)] = q_value
        # f.write(
        #     f"{hand_str},{q_values_dict['CHECK']},{q_values_dict['FOLD']},{q_values_dict['CALL']},{q_values_dict['RAISE']}\n"
        # )
    for state_action in q_table.keys():
        state, action = state_action
        print(f"State: {state}", file=save_file)
        print(f"Action: {action}", file=save_file)
        print(f"Q-value: {q_table[state_action]}", file=save_file)
        print("", file=save_file)


def main():
    action_space = [Actions.CHECK, Actions.FOLD, Actions.CALL, Actions.RAISE]
    agent1 = DQNAgent(action_space=action_space)
    agent2 = DQNAgent(action_space=action_space)

    for epoch in range(0, 500):  # Adjust the number of epochs as needed
        epoch_reward = 0
        for _ in tqdm(range(10000)):
            reward = play_game(agent1, agent2)
            epoch_reward += reward
        print(
            f"Epoch {epoch}, Total Reward: {epoch_reward}, BB per hand: {epoch_reward / 10000}"
        )

        if epoch % 10 == 0 and epoch > 0:
            agent1.save_model(f"agent1_model_epoch_{epoch}.pth")
            agent2.save_model(f"agent2_model_epoch_{epoch}.pth")

            with open(f"q_table_agent1_{epoch}.txt", "w", encoding="utf-8") as file:
                # agent1.print_table(save_file=file)
                calculate_q_values_for_starting_hands(
                    agent1, save_file=file, player1=True
                )
            with open(f"q_table_agent2_{epoch}.txt", "w", encoding="utf-8") as file:
                # agent2.print_table(save_file=file)
                calculate_q_values_for_starting_hands(
                    agent2, save_file=file, player1=False
                )

            # calculate_q_values_for_starting_hands(
            #     agent1, "agent1_starting_hand_q_values.csv", player1=True
            # )
            # calculate_q_values_for_starting_hands(
            #     agent2, "agent2_starting_hand_q_values.csv", player1=False
            # )

    total_reward = 0
    for i in range(1000):
        if i % 50 == 0:
            print(f"Simulation game {i}")
        total_reward += simulate_game(agent1, agent2)

    print(f"Average reward over 1000 simulated games: {total_reward / 1000}")


if __name__ == "__main__":
    main()
    # print(get_all_possible_states(True))

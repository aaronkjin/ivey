import tkinter as tk
from tkinter import ttk
from collections import defaultdict
from enum import Enum
import numpy as np


class Actions(Enum):
    CHECK = 0
    FOLD = 1
    CALL = 2
    RAISE = 3


def standardize_hand_format(hand):
    """Convert hand format to match Q-table format (e.g., 'AKs' -> 'AKS')"""
    # Handle pairs (e.g., 'AA' -> 'AAO')
    if len(hand) == 2 and hand[0] == hand[1]:
        return hand + "O"
    # Handle suited hands
    elif hand.endswith("s"):
        return hand[:-1] + "S"
    # Handle offsuit hands
    elif hand.endswith("o"):
        return hand[:-1] + "O"
    return hand


def load_table(filename):
    q_table = defaultdict(float)

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
                action = Actions[action_str.split(".")[-1]]
                q_table[(state, action)] = q_value

    print(f"Q-table successfully loaded from '{filename}'.")

    return q_table


def load_action_table(q_table, state_pattern):
    action_table = {}
    for state, action in q_table.keys():
        cards = state.split(",")[0]
        state_actions = ",".join(state.split(",")[1:])
        if state_actions == state_pattern:
            if cards not in action_table:
                action_table[cards] = []
            action_table[cards].append((action, q_table[(state, action)]))

    return action_table


def create_poker_chart(filename, state_pattern=""):
    q_table = load_table(filename)
    action_table = load_action_table(q_table, state_pattern)

    title = "Preflop Poker Strategy Chart"
    legend_title = "Legend"
    if state_pattern == "":
        title += " Agent 1 First Action"
        legend_items = [
            ("Raise", Actions.RAISE),
            ("Call", Actions.CALL),
            ("Fold", Actions.FOLD),
        ]
    elif state_pattern == "C,R":
        title += " Agent 1 vs Raise"
        legend_items = [
            ("Call", Actions.CALL),
            ("Fold", Actions.FOLD),
        ]
    elif state_pattern == "C":
        title += " Agent 2 vs Call"
        legend_items = [
            ("Check", Actions.CHECK),
            ("Raise", Actions.RAISE),
        ]
    elif state_pattern == "R":
        title += " Agent 2 vs Raise"
        legend_items = [
            ("Call", Actions.CALL),
            ("Fold", Actions.FOLD),
        ]

    root = tk.Tk()

    root.title(title)
    root.configure(bg="#2d2d2d")

    grid_rows = [
        "AA AKs AQs AJs ATs A9s A8s A7s A6s A5s A4s A3s A2s",
        "AKo KK KQs KJs KTs K9s K8s K7s K6s K5s K4s K3s K2s",
        "AQo KQo QQ QJs QTs Q9s Q8s Q7s Q6s Q5s Q4s Q3s Q2s",
        "AJo KJo QJo JJ JTs J9s J8s J7s J6s J5s J4s J3s J2s",
        "ATo KTo QTo JTo TT T9s T8s T7s T6s T5s T4s T3s T2s",
        "A9o K9o Q9o J9o T9o 99 98s 97s 96s 95s 94s 93s 92s",
        "A8o K8o Q8o J8o T8o 98o 88 87s 86s 85s 84s 83s 82s",
        "A7o K7o Q7o J7o T7o 97o 87o 77 76s 75s 74s 73s 72s",
        "A6o K6o Q6o J6o T6o 96o 86o 76o 66 65s 64s 63s 62s",
        "A5o K5o Q5o J5o T5o 95o 85o 75o 65o 55 54s 53s 52s",
        "A4o K4o Q4o J4o T4o 94o 84o 74o 64o 54o 44 43s 42s",
        "A3o K3o Q3o J3o T3o 93o 83o 73o 63o 53o 43o 33 32s",
        "A2o K2o Q2o J2o T2o 92o 82o 72o 62o 52o 42o 32o 22",
    ]

    action_colors = {
        Actions.RAISE: "#FF4444",  # Red
        Actions.CALL: "#44FF44",  # Green
        Actions.FOLD: "#0000FF",  # Blue
        Actions.CHECK: "#44FF44",  # Green
    }

    frame = ttk.Frame(root)
    frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))

    for i in range(13):
        root.rowconfigure(i, weight=1)
        root.columnconfigure(i, weight=1)

    for i, row in enumerate(grid_rows):
        hands = row.split()
        for j, hand in enumerate(hands):
            lookup_hand = standardize_hand_format(hand)

            hand_data = action_table[lookup_hand]
            q_vals = [val for (_, val) in hand_data]
            best_action_idx = np.argmax(q_vals)
            best_action = hand_data[best_action_idx][0]
            best_q_val = hand_data[best_action_idx][1]
            text = f"{hand}\n{best_q_val:.2f}"

            color = action_colors[best_action]

            label = tk.Label(
                frame,
                text=text,
                bg=color,
                fg="white",
                borderwidth=1,
                relief="solid",
                width=6,
                height=2,
            )
            label.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)

    legend_frame = ttk.Frame(root)
    legend_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)

    ttk.Label(legend_frame, text=legend_title, font=("Arial", 12, "bold")).grid(
        row=0, column=0, sticky=tk.W
    )

    for i, (text, action) in enumerate(legend_items):
        label = tk.Label(
            legend_frame,
            text=text,
            bg=action_colors[action],
            fg="black",
            width=10,
            relief="solid",
        )
        label.grid(row=i + 1, column=0, padx=5, pady=2, sticky=tk.W)

    root.mainloop()


create_poker_chart(filename="q_table_agent1_1880.txt", state_pattern="")
create_poker_chart(filename="q_table_agent1_1990.txt", state_pattern="C,R")
create_poker_chart(filename="q_table_agent2_1990.txt", state_pattern="C")
create_poker_chart(filename="q_table_agent2_1990.txt", state_pattern="R")

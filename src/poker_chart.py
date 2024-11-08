import tkinter as tk
from tkinter import ttk
import os

def standardize_hand_format(hand):
    """Convert hand format to match Q-table format (e.g., 'AKs' -> 'AKS')"""
    # Handle pairs (e.g., 'AA' -> 'AAO')
    if len(hand) == 2 and hand[0] == hand[1]:
        return hand + 'O'
    # Handle suited hands
    elif hand.endswith('s'):
        return hand[:-1] + 'S'
    # Handle offsuit hands
    elif hand.endswith('o'):
        return hand[:-1] + 'O'
    return hand

def load_q_table(filename, is_agent2=False, vs_call=False):
    """Load and process Q-table data from file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    filepath = os.path.join(parent_dir, filename)
    
    print(f"Attempting to load Q-table from: {filepath}")
    
    actions = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state = None
            action = None
            for line in f:
                line = line.strip()
                if line.startswith("State: "):
                    state_line = line.replace("State: ", "")
                    # For Agent 2 facing a call, only look at states ending in ",C"
                    if is_agent2 and vs_call and not state_line.endswith(",C"):
                        state = None
                        continue
                    state = state_line.split(",")[0]  # Get just the hand part
                    if state not in actions:
                        actions[state] = {"values": {}}
                elif line.startswith("Action: "):
                    action_str = line.replace("Action: ", "")
                    if is_agent2 and vs_call:
                        # Only consider CHECK and RAISE actions when Agent 2 faces a call
                        if action_str == "Actions.CHECK":
                            action = "CHECK"
                        elif action_str == "Actions.RAISE":
                            action = "RAISE"
                        else:
                            action = None
                    else:
                        # Normal action processing for other cases
                        if action_str == "Actions.CALL":
                            action = "CALL"
                        elif action_str == "Actions.FOLD":
                            action = "FOLD"
                        elif action_str == "Actions.RAISE":
                            action = "RAISE"
                elif line.startswith("Q-value: ") and state and action:
                    q_value = float(line.replace("Q-value: ", ""))
                    if action not in actions[state]["values"] or q_value > actions[state]["values"][action]:
                        actions[state]["values"][action] = q_value
                    action = None

        # Determine best action for each hand
        for hand in actions:
            values = actions[hand]["values"]
            if values:
                best_action = max(values.items(), key=lambda x: x[1])[0]
                actions[hand]["best_action"] = best_action
                actions[hand]["q_value"] = max(values.values())
                
        print(f"\nFirst few hands and their optimal actions:")
        for i, (hand, data) in enumerate(list(actions.items())[:5]):
            print(f"{hand}:")
            print(f"  Values: {data['values']}")
            print(f"  Best action: {data.get('best_action', 'UNKNOWN')}")
            print(f"  Q-value: {data.get('q_value', 0.0):.2f}")
                
        return actions
                
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        return {}
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return {}

def create_poker_chart(is_agent2=False, vs_call=False):
    root = tk.Tk()
    title = "Preflop Poker Strategy Chart"
    if is_agent2:
        title += " (Agent 2"
        if vs_call:
            title += " vs Call)"
        else:
            title += " vs Raise)"
    root.title(title)
    root.configure(bg='#2d2d2d')

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
        "A2o K2o Q2o J2o T2o 92o 82o 72o 62o 52o 42o 32o 22"
    ]

    hand_actions = load_q_table("q_table_agent2_490.txt", is_agent2, vs_call)
    print(f"\nTotal hands processed: {len(hand_actions)}")

    action_colors = {
        "RAISE": "#FF4444",  # Red
        "CALL": "#44FF44",   # Green
        "FOLD": "#0000FF",   # Blue
        "CHECK": "#44FF44",  # Green
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
            
            hand_data = hand_actions.get(lookup_hand, {})
            best_action = hand_data.get("best_action", "CHECK" if is_agent2 and vs_call else "FOLD")
            q_value = hand_data.get("q_value", 0.0)
            
            if 'values' in hand_data:
                text = f"{hand}\n{q_value:.2f}"
            else:
                text = hand
            
            color = action_colors[best_action]
            
            label = tk.Label(frame, text=text, bg=color, fg='white',
                           borderwidth=1, relief="solid", width=6, height=2)
            label.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)

    legend_frame = ttk.Frame(root)
    legend_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)

    legend_title = "Legend"
    if is_agent2:
        legend_title += " (Agent 2"
        if vs_call:
            legend_title += " vs Call):"
            legend_items = [
                ("Raise", "RAISE"),
                ("Check", "CHECK")
            ]
        else:
            legend_title += " vs Raise):"
            legend_items = [
                ("Call", "CALL"),
                ("Fold", "FOLD")
            ]
    else:
        legend_title += ":"
        legend_items = [
            ("Raise", "RAISE"),
            ("Call", "CALL"),
            ("Fold", "FOLD")
        ]

    ttk.Label(legend_frame, text=legend_title, 
             font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W)

    for i, (text, action) in enumerate(legend_items):
        label = tk.Label(legend_frame, text=text, bg=action_colors[action],
                        fg='black', width=10, relief="solid")
        label.grid(row=i+1, column=0, padx=5, pady=2, sticky=tk.W)

    root.mainloop()

if __name__ == "__main__":
    create_poker_chart(is_agent2=True, vs_call=True)
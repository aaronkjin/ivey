from phevaluator import evaluate_cards
import numpy as np
import random
from collections import defaultdict


def all_cards():
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["c", "d", "h", "s"]
    cards = []
    for rank in ranks:
        for suit in suits:
            cards.append(f"{rank}{suit}")

    return cards


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


def draw_cards_for_hand_type(hand_type, cards):

    card1 = hand_type[0]
    card2 = hand_type[1]
    suited = hand_type[2] == "S"
    paired = card1 == card2

    hand = []
    for card in cards:
        if len(hand) == 0:
            if card[0] == card1 or card[0] == card2:
                hand.append(card)
        elif card[0] != hand[0][0]:
            if card[0] == card1 or card[0] == card2:
                if (card[1] == hand[0][1]) == suited:
                    hand.append(card)
                    break
        elif paired:
            hand.append(card)
            break

    for card in hand:
        cards.remove(card)

    return hand


def calc_ev_for_matchup(hand_type1, hand_type2):
    hand1_wins = 0.0
    hand1_ties = 0.0
    num_iters = 1000
    for _ in range(num_iters):
        cards = all_cards()
        np.random.shuffle(cards)
        hand1 = draw_cards_for_hand_type(hand_type1, cards)
        hand2 = draw_cards_for_hand_type(hand_type2, cards)

        board = random.sample(cards, 5)
        hand1_score = evaluate_cards(*(board + hand1))
        hand2_score = evaluate_cards(*(board + hand2))
        if hand1_score < hand2_score:
            hand1_wins += 1
        elif hand1_score == hand2_score:
            hand1_ties += 1

    hand1_equity = hand1_wins / num_iters + hand1_ties / (num_iters * 2)
    return hand1_equity


if __name__ == "__main__":
    starting_hands = generate_starting_hands()
    ev_dict = defaultdict(lambda: defaultdict(float))
    for i in range(len(starting_hands)):
        for j in range(len(starting_hands)):
            hand1 = starting_hands[i]
            hand2 = starting_hands[j]
            if hand1 == hand2:
                ev_dict[hand1][hand2] = 0.5
                continue

            if hand2 in ev_dict:
                if hand1 in ev_dict[hand2]:
                    ev_dict[hand1][hand2] = 1.0 - ev_dict[hand2][hand1]
                    continue

            ev_dict[hand1][hand2] = calc_ev_for_matchup(hand1, hand2)

            print(hand1, hand2, ev_dict[hand1][hand2])

    import json

    with open("preflop_evs.json", "w") as f:
        json.dump(ev_dict, f)

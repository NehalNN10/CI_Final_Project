import enum
import random
import numpy as np
from collections import defaultdict

# Constants
RED = "r"
BLU = "b"
JOKER = "#"
EMPTY = "_"

# Enum for winners
Winner = enum.Enum("Winner", "red blue draw")
Player = enum.Enum("Player", "red blue")

# Card to board coordinates mapping (must be defined)
COORDS = defaultdict(list)
# Populate this with actual card positions from Sequence board
# Example: COORDS['as'] = [(0,0), (2,2)] etc.


class AgentState:
    def __init__(self, _id):
        self.id = _id
        self.color = BLU if _id == 0 else RED
        self.hand = []
        self.completed_seqs = 0
        self.score = 0


class SequenceState:
    def __init__(self):
        # self.board = np.zeros((10, 10))  # 10x10 board
        self.board = [
            ["jk", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "jk"],
            ["6c", "5c", "4c", "3c", "2c", "ah", "kh", "qh", "th", "ts"],
            ["7c", "as", "2d", "3d", "4d", "5d", "6d", "7d", "9h", "qs"],
            ["8c", "ks", "6c", "5c", "4c", "3c", "2c", "8d", "8h", "ks"],
            ["9c", "qs", "7c", "6h", "5h", "4h", "ah", "9d", "7h", "as"],
            ["tc", "ts", "8c", "7h", "2h", "3h", "kh", "td", "6h", "2d"],
            ["qc", "9s", "9c", "8h", "9h", "th", "qh", "qd", "5h", "3d"],
            ["kc", "8s", "tc", "qc", "kc", "ac", "ad", "kd", "4h", "4d"],
            ["ac", "7s", "6s", "5s", "4s", "3s", "2s", "2h", "3h", "5d"],
            ["jk", "ad", "kd", "qd", "td", "9d", "8d", "7d", "6d", "jk"],
        ]
        self.hand = np.zeros(104)  # One-hot vector for player's hand
        self.opponent_belief = np.full(104, 1 / 104)  # Uniform prior
        self.chips = np.zeros((10, 10))  # -1: red, 0: empty, 1: blue
        self.discard_pile = []
        self.current_player = 0
        self.turn = 0

    def get_observation(self):
        """Returns observation for RL agent"""
        return {
            "board": self.board,
            "chips": self.chips,
            "hand": self.hand,
            "opponent_belief": self.opponent_belief,
            "current_player": self.current_player,
        }

    def update_belief(self, card_played):
        """Bayesian update of opponent card beliefs"""
        if card_played in self.discard_pile:
            self.opponent_belief[card_played] = 0
        # Normalize probabilities
        total = np.sum(self.opponent_belief)
        if total > 0:
            self.opponent_belief /= total


class Deck:
    def __init__(self):
        self.cards = []
        self.reset()

    def reset(self):
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "t", "j", "q", "k", "a"]
        suits = ["d", "c", "h", "s"]
        self.cards = [r + s for r in ranks for s in suits] * 2
        random.shuffle(self.cards)

    def deal(self, n=7):
        return [self.cards.pop() for _ in range(n)] if n <= len(self.cards) else []


class SequenceEnv:
    def __init__(self):
        self.state = SequenceState()
        self.reset()

    def reset(self):
        """Initialize new game"""
        self.state = SequenceState()
        self.deck = Deck()

        # Initialize agents
        self.agents = [AgentState(i) for i in range(2)]
        for agent in self.agents:
            agent.hand = self.deck.deal(7)
            # Update one-hot hand representation
            for card in agent.hand:
                self.state.hand[card_to_index(card)] = 1

        # Initialize board positions
        self.state.chips = np.zeros((10, 10))
        for r, c in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            self.state.chips[r][c] = JOKER

        self.done = False
        self.winner = None
        return self.state.get_observation()

    def step(self, action):
        """Execute one game turn"""
        if self.done:
            return self.state.get_observation(), 0, True, {}

        player = self.agents[self.state.current_player]
        reward = 0

        # Execute action
        if action["type"] == "place":
            r, c = action["coords"]
            self.state.chips[r][c] = 1 if player.color == BLU else -1
            self.remove_card(player, action["card"])

        elif action["type"] == "remove":
            r, c = action["coords"]
            self.state.chips[r][c] = EMPTY
            self.remove_card(player, action["card"])

        # Update game state
        self.state.discard_pile.append(action["card"])
        self.state.update_belief(action["card"])
        self.check_win_conditions()

        # Switch player
        self.state.current_player = 1 - self.state.current_player
        self.state.turn += 1

        # Calculate reward
        if self.done:
            reward = (
                1
                if (self.winner == Winner.blue and player.id == 0)
                or (self.winner == Winner.red and player.id == 1)
                else -1
            )
        else:
            # Intermediate reward for sequences
            reward = player.completed_seqs * 0.1

        return self.state.get_observation(), reward, self.done, {}

    def remove_card(self, player, card):
        """Remove card from player's hand and draw new one"""
        player.hand.remove(card)
        self.state.hand[card_to_index(card)] = 0
        if self.deck.cards:
            new_card = self.deck.deal(1)[0]
            player.hand.append(new_card)
            self.state.hand[card_to_index(new_card)] = 1

    def check_win_conditions(self):
        """Check if game has been won"""
        for player in self.agents:
            player.completed_seqs = self.count_sequences(player.color)

        if self.agents[0].completed_seqs >= 2:
            self.done = True
            self.winner = Winner.blue
        elif self.agents[1].completed_seqs >= 2:
            self.done = True
            self.winner = Winner.red
        elif not self.deck.cards and all(len(p.hand) == 0 for p in self.agents):
            self.done = True
            self.winner = Winner.draw

    def count_sequences(self, color):
        count = 0
        # Check all possible 5-in-a-row sequences

        # Check rows
        for r in range(10):
            for c in range(6):
                if all(self.state.chips[r][c + i] == color for i in range(5)):
                    count += 1
                elif all(
                    self.state.chips[r][c + i] == JOKER
                    or self.state.chips[r][c + i] == color
                    for i in range(5)
                ):
                    count += 1

        # Check columns
        for r in range(6):
            for c in range(10):
                if all(self.state.chips[r][c + i] == color for i in range(5)):
                    count += 1
                elif all(
                    self.state.chips[r + i][c] == JOKER
                    or self.state.chips[r + i][c] == color
                    for i in range(5)
                ):
                    count += 1

        # Check diagonals
        for r in range(6):
            for c in range(6):
                if all(self.state.chips[r + i][c + i] == color for i in range(5)):
                    count += 1
                elif all(
                    self.state.chips[r + i][c + i] == JOKER
                    or self.state.chips[r + i][c + i] == color
                    for i in range(5)
                ):
                    count += 1
                if all(self.state.chips[r + 4 - i][c + i] == color for i in range(5)):
                    count += 1
                elif all(
                    self.state.chips[r + 4 - i][c + i] == JOKER
                    or self.state.chips[r + 4 - i][c + i] == color
                    for i in range(5)
                ):
                    count += 1

        return count

    def legal_actions(self):
        """Generate all legal moves for current player"""
        actions = []
        player = self.agents[self.state.current_player]

        for card in player.hand:
            if card in ["jd", "jc"]:  # Two-eyed jacks (wild)
                for r in range(10):
                    for c in range(10):
                        if self.state.chips[r][c] == EMPTY:
                            actions.append(
                                {"type": "place", "card": card, "coords": (r, c)}
                            )
            elif card in ["jh", "js"]:  # One-eyed jacks (remove)
                opponent = BLU if player.color == RED else RED
                for r in range(10):
                    for c in range(10):
                        if self.state.chips[r][c] == opponent:
                            actions.append(
                                {"type": "remove", "card": card, "coords": (r, c)}
                            )
            else:  # Regular cards
                for r, c in COORDS[card]:
                    if self.state.chips[r][c] == EMPTY:
                        actions.append(
                            {"type": "place", "card": card, "coords": (r, c)}
                        )
        return actions


def card_to_index(card):
    """Convert card string to index in one-hot vector"""
    rank_map = {
        "2": 0,
        "3": 1,
        "4": 2,
        "5": 3,
        "6": 4,
        "7": 5,
        "8": 6,
        "9": 7,
        "t": 8,
        "j": 9,
        "q": 10,
        "k": 11,
        "a": 12,
    }
    suit_map = {"d": 0, "c": 1, "h": 2, "s": 3}

    if len(card) != 2:
        return -1  # Invalid card

    rank, suit = card[0], card[1]
    return rank_map[rank] * 4 + suit_map[suit]

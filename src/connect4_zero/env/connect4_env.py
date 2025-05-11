import enum
import random
import numpy as np

# Constants
RED = "r"
BLU = "b"
JOKER = "#"
EMPTY = "_"

# Enum for winners
Winner = enum.Enum("Winner", "red blue draw")
Player = enum.Enum("Player", "red blue")


class AgentState:
    def __init__(self, _id):
        self.id = _id
        self.color = BLU if _id == 0 else RED  # Player 0 is blue, player 1 is red
        self.hand = []
        self.completed_seqs = 0
        self.traded = False


class Deck:
    def __init__(self):
        self.cards = []
        self.reset()

    def reset(self):
        # Sequence uses 2 standard decks (no jokers)
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "t", "j", "q", "k", "a"]
        suits = ["d", "c", "h", "s"]
        self.cards = [r + s for r in ranks for s in suits] * 2
        random.shuffle(self.cards)

    def deal(self, n=1):
        return [self.cards.pop() for _ in range(n)]


class SequenceEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize board
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
        self.chips = [[EMPTY for _ in range(10)] for _ in range(10)]

        # Set up joker spaces
        for r, c in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            self.chips[r][c] = JOKER

        # Initialize deck and deal cards
        self.deck = Deck()
        self.agents = [AgentState(i) for i in range(2)]  # Only 2 players now
        for agent in self.agents:
            agent.hand = self.deck.deal(6)

        # Initial draft cards
        self.draft_cards = self.deck.deal(5)

        self.current_player = 0
        self.turn = 0
        self.done = False
        self.winner = None
        return self

    def player_turn(self):
        return self.agents[self.current_player]

    def step(self, action):
        if action is None:  # Resign
            self._resigned()
            return self.chips, {}

        player = self.player_turn()

        # Handle different action types
        if action["type"] == "place":
            r, c = action["coords"]
            self.chips[r][c] = player.color
            player.hand.remove(action["card"])

        elif action["type"] == "remove":
            r, c = action["coords"]
            self.chips[r][c] = EMPTY
            player.hand.remove(action["card"])

        elif action["type"] == "trade":
            player.hand.remove(action["card"])
            player.traded = True

        # Replace played card with draft card
        if action["type"] != "trade":
            player.hand.append(action["draft"])
            self.draft_cards.remove(action["draft"])
            self.draft_cards.extend(self.deck.deal(1))
            player.traded = False
            self.current_player = (self.current_player + 1) % 2  # Only 2 players now
            self.turn += 1

        # Check for winning sequences
        self.check_for_sequences()

        return self.chips, {}

    def check_for_sequences(self):
        # Check sequences for both players
        for player in self.agents:
            player.completed_seqs = self.count_sequences(player.color)

        # Check win conditions
        if self.agents[0].completed_seqs >= 2:
            self.done = True
            self.winner = Winner.blue
        elif self.agents[1].completed_seqs >= 2:
            self.done = True
            self.winner = Winner.red
        elif len(self.deck.cards) == 0:  # Deck exhausted
            self.done = True
            self.winner = Winner.draw

    def count_sequences(self, color):
        count = 0
        # Check all possible 5-in-a-row sequences
        # (Implementation would be similar to your check_for_fives but return count)
        return count

    def legal_moves(self):
        actions = []
        player = self.player_turn()

        # Check for dead cards that can be traded
        if not player.traded:
            for card in player.hand:
                if card[0] != "j" and not any(
                    self.chips[r][c] == EMPTY for r, c in COORDS[card]
                ):
                    for draft in self.draft_cards:
                        actions.append({"type": "trade", "card": card, "draft": draft})

        # Regular moves
        for card in player.hand:
            if card in ["jd", "jc"]:  # Two-eyed jacks (place anywhere)
                for r in range(10):
                    for c in range(10):
                        if self.chips[r][c] == EMPTY:
                            for draft in self.draft_cards:
                                actions.append(
                                    {
                                        "type": "place",
                                        "card": card,
                                        "coords": (r, c),
                                        "draft": draft,
                                    }
                                )
            elif card in ["jh", "js"]:  # One-eyed jacks (remove opponent)
                opponent_color = BLU if player.color == RED else RED
                for r in range(10):
                    for c in range(10):
                        if self.chips[r][c] == opponent_color:
                            for draft in self.draft_cards:
                                actions.append(
                                    {
                                        "type": "remove",
                                        "card": card,
                                        "coords": (r, c),
                                        "draft": draft,
                                    }
                                )
            else:  # Regular cards
                for r, c in COORDS[card]:
                    if self.chips[r][c] == EMPTY:
                        for draft in self.draft_cards:
                            actions.append(
                                {
                                    "type": "place",
                                    "card": card,
                                    "coords": (r, c),
                                    "draft": draft,
                                }
                            )
        return actions

    def _resigned(self):
        current_color = self.agents[self.current_player].color
        self.winner = Winner.blue if current_color == BLU else Winner.red
        self.done = True

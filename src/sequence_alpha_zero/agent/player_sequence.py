from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
from logging import getLogger
import asyncio

import numpy as np

from connect4_zero.agent.api_connect4 import Connect4ModelAPI
from connect4_zero.config import Config
from connect4_zero.env.sequence_env import Connect4Env, Winner, Player

CounterKey = namedtuple("CounterKey", "board next_player")
QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")

logger = getLogger(__name__)


class Connect4Player:
    def __init__(self, config: Config, model, play_config=None):

        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = Connect4ModelAPI(self.config, self.model)

        self.labels_n = config.n_labels
        self.var_n = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_w = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_q = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_u = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_p = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.expanded = set()
        self.now_expanding = set()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

        self.thinking_history = {}  # for fun

    def action(self, board):

        env = Connect4Env().update(board)
        key = self.counter_key(env)

        for tl in range(self.play_config.thinking_loop):
            if tl > 0 and self.play_config.logging_thinking:
                logger.debug(f"continue thinking: policy move=({action % 8}, {action // 8}), "
                             f"value move=({action_by_value % 8}, {action_by_value // 8})")
            self.search_moves(board)
            policy = self.calc_policy(board)
            action = int(np.random.choice(range(self.labels_n), p=policy))
            action_by_value = int(np.argmax(self.var_q[key] + (self.var_n[key] > 0)*100))
            if action == action_by_value or env.turn < self.play_config.change_tau_turn:
                break

        # this is for play_gui, not necessary when training.
        self.thinking_history[env.observation] = HistoryItem(action, policy, list(self.var_q[key]), list(self.var_n[key]))

        self.moves.append([env.observation, list(policy)])
        return action

    def ask_thought_about(self, board) -> HistoryItem:
        return self.thinking_history.get(board)

    def search_moves(self, board):
        loop = self.loop
        self.running_simulation_num = 0

        coroutine_list = []
        for it in range(self.play_config.simulation_num_per_move):
            cor = self.start_search_my_move(board)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    async def start_search_my_move(self, board):
        self.running_simulation_num += 1
        with await self.sem:  # reduce parallel search number
            env = Connect4Env().update(board)
            leaf_v = await self.search_my_move(env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v

    async def search_my_move(self, env: Connect4Env, is_root_node=False):
        """

        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)
        :param env:
        :param is_root_node:
        :return:
        """
        if env.done:
            if env.winner == Winner.white:
                return 1
            elif env.winner == Winner.black:
                return -1
            else:
                return 0

        key = self.counter_key(env)

        while key in self.now_expanding:
            await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)

        # is leaf?
        if key not in self.expanded:  # reach leaf node
            leaf_v = await self.expand_and_evaluate(env)
            if env.player_turn() == Player.white:
                return leaf_v  # Value for white
            else:
                return -leaf_v  # Value for white == -Value for white

        action_t = self.select_action_q_and_u(env, is_root_node)
        _, _ = env.step(action_t)

        virtual_loss = self.config.play.virtual_loss
        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss
        leaf_v = await self.search_my_move(env)  # next move

        # on returning search path
        # update: N, W, Q, U
        n = self.var_n[key][action_t] = self.var_n[key][action_t] - virtual_loss + 1
        w = self.var_w[key][action_t] = self.var_w[key][action_t] + virtual_loss + leaf_v
        self.var_q[key][action_t] = w / n
        return leaf_v

    async def expand_and_evaluate(self, env):
        """expand new leaf

        update var_p, return leaf_v

        :param ChessEnv env:
        :return: leaf_v
        """
        key = self.counter_key(env)
        self.now_expanding.add(key)

        black_ary, white_ary = env.black_and_white_plane()
        state = [black_ary, white_ary] if env.player_turn() == Player.black else [white_ary, black_ary]
        future = await self.predict(np.array(state))  # type: Future
        await future
        leaf_p, leaf_v = future.result()

        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.

        speed up about 45sec -> 15sec for example.
        :return:
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.play.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            # logger.debug(f"predicting {len(item_list)} items")
            data = np.array([x.state for x in item_list])
            policy_ary, value_ary = self.api.predict(data)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

    def finish_game(self, z):
        """

        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

    def calc_policy(self, board):
        """calc π(a|s0)
        :return:
        """
        pc = self.play_config
        env = Connect4Env().update(board)
        key = self.counter_key(env)
        if env.turn < pc.change_tau_turn:
            return self.var_n[key] / np.sum(self.var_n[key])  # tau = 1
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(self.labels_n)
            ret[action] = 1
            return ret

    @staticmethod
    def counter_key(env: Connect4Env):
        return CounterKey(env.observation, env.turn)

    def select_action_q_and_u(self, env, is_root_node):
        key = self.counter_key(env)

        legal_moves = env.legal_moves()

        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        p_ = self.var_p[key]

        if is_root_node:  # Is it correct?? -> (1-e)p + e*Dir(0.03)
            p_ = (1 - self.play_config.noise_eps) * p_ + \
                 self.play_config.noise_eps * np.random.dirichlet([self.play_config.dirichlet_alpha] * self.labels_n)

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])
        if env.player_turn() == Player.white:
            v_ = (self.var_q[key] + u_ + 1000) * legal_moves
        else:
            # When enemy's selecting action, flip Q-Value.
            v_ = (-self.var_q[key] + u_ + 1000) * legal_moves

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t

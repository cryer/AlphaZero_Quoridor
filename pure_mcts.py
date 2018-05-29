# -*- coding: utf-8 -*-
import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(game):
    # 得到一个合法动作空间大小的随机概率分布
    action_probs = np.random.rand(len(game.actions()))
    return zip(game.actions(), action_probs)


def policy_value_fn(game):
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(game.actions())) / len(game.actions())
    return zip(game.actions(), action_probs), 0


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, game):
        node = self._root
        while (1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            game.step(action)

        action_probs, _ = self._policy(game)
        # Check for end of game
        end, winner = game.has_a_winner()
        if not end:
            node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(game)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    # 随机模拟下去，获得胜负结果，但是最深不模拟超过1000层
    def _evaluate_rollout(self, game, limit=1000):
        player = game.get_current_player()
        for i in range(limit):
            end, winner = game.has_a_winner()
            # print("end:",end,"winnner",winner)
            if end:
                break
            elif i == (limit-1):
                print("WARNING: rollout reached move limit")
                break
            # action_probs 是 zip(game.actions(), action_probs)
            action_probs = rollout_policy_fn(game)
            # 所以itemgetter(1)，获取第二个数值，也就是随机概率分布，选择最大的
            # 返回的依然是动作和概率元组对，选择第一个动作
            max_action = max(action_probs, key=itemgetter(1))[0]
            game.step(max_action)
        # else:
        #     print("WARNING: rollout reached move limit")
        if winner == None:
            print("没有模拟出胜者")
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, game):
        for n in range(self._n_playout):
            game_copy = copy.deepcopy(game)
            self._playout(game_copy)
            print("模拟一次结束")
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, c_puct=5, n_playout=50):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def choose_action(self, game):
        sensible_moves = game.actions()
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(game)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

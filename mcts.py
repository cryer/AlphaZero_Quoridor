import numpy as np
import copy


from constant import *
from functools import reduce

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
    """

    def __init__(self, parent, prior_p, state, action):
        self._parent = parent
        self._children = {}  #
        self._n_visits = 0
        self._state = state
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._action = action
        # self._game = game

    def expand(self, action_priors, game):
        duplicated_node = False
        parent_node = None
        parent_state = None

        #zip_length = 0
        #action_priors = list(action_priors)
        #noise_prob = np.random.dirichlet(0.3 * np.ones(len(action_priors)))

        for action, prob in action_priors:
            """
            if action < 12:

                # Code for restrict dummy expand

                duplicated_node = False

                # copy game - step action - get state after step(action) end
                c_game = copy.deepcopy(game)
                c_game.step(action)
                next_state = c_game.state()

                # if 'self' is not root node
                if self._parent is not None:
                    parent_node = self._parent  # get parent node
                    parent_state = parent_node._state  # get parent node state

                # Compare all states in nodes and next state
                while parent_node is not None:
                    if np.array_equal(parent_state, next_state):
                        duplicated_node = True
                        break
                    else:
                        # get parent-parent node and parent-parent node state
                        parent_node = parent_node._parent
                        if parent_node is not None:
                            parent_state = parent_node._state
        
            if not duplicated_node and action not in self._children:
                self._children[action] = TreeNode(self, prob, next_state, action)
            """
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, None, action)

    def select(self, c_puct):
        """
        """

        #if np.random.random_sample() < 0.7:
        #    return reduce(lambda x, y: x if (x[0] < 2 and y[0] >= 2) else x if ((x[0] < 2 and y[0] < 2) and (x[1].get_value(c_puct) > y[1].get_value(c_puct))) else x if (x[1].get_value(c_puct) > y[1].get_value(c_puct)) else y, self._children.items())

        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        """
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, reward):


        if self._parent:
            self._parent.update_recursive(-reward)

        self.update(reward)

    def get_value(self, c_puct):
        """
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)

        return self._Q + self._u

    def is_leaf(self):
        """
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def get_parent(self):
        return self._parent


class MCTS(object):
    """
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1800):
        """
        """
        self._root = TreeNode(None, 1.0, None, None)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    # Fix : get current_player param info when the first simulation started.
    def _playout(self, game, current_player):
        """
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break

            action, node = node.select(self._c_puct)
            game.step(action)  #
        # state = game.state()

        action_probs, leaf_value = self._policy(game)
        end, winner = game.has_a_winner()


        if not end:
            # Add an incompleted code to make pawn avoid dead-end section.
            """
            if np.sum(game.actions()[:4]) <= 1:
                leaf_value = -1.0 if game.get_current_player == current_player else 1.0
            else:
            """
            node.expand(action_probs, game)
        else:
            leaf_value = 1.0 if winner == game.get_current_player() else -1.0  # Fix bug that all winners are current player
            # print(leaf_value)
        # print("call update")

        node.update_recursive(-leaf_value)

    def get_move_probs(self, game, temp=1e-3, time_step=0):
        """
        """
        for n in range(self._n_playout):
            game_copy = copy.deepcopy(game)
            # state = game.state()
            # state_copy = copy.deepcopy(state)
            self._playout(game_copy, game_copy.get_current_player())

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        visits = np.array(visits)

        """
        if time_step < TAU_THRES:
            act_probs = visits / visits.sum()
        else:
            act_probs = np.zeros(len(visits))
            max_idx = np.argwhere(visits == visits.max())

            action_index = max_idx[np.random.choice(len(max_idx))]
            act_probs[action_index] = 1
        """

        # q_vals = [node._Q for act, node in self._root._children.items()]
        # print("-" * 30)
        # print("q_vals : ", q_vals)
        # print("-" * 30)
        return acts, act_probs

    def update_with_move(self, last_move, state):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0, state, last_move)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    #
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=1):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    #
    def set_player_ind(self, p):
        self.player = p

    #
    def reset_player(self):
        self.mcts.update_with_move(-1, None)

    # Choose an action during the play
    def choose_action(self, game, temp=1e-3, return_prob=0, time_step=0):
        sensible_moves = game.actions()  # 获取所有可行的落子
        move_probs = np.zeros(12 + (BOARD_SIZE - 1) ** 2 * 2)  # 获取落子的概率，由神经网络输出

        if len(sensible_moves) > 0:  # 棋盘未耗尽时
            acts, probs = self.mcts.get_move_probs(game, temp, time_step)  # 获取落子以及对应的落子概率
            move_probs[list(acts)] = probs  # 将概率转到move_probs列表中
            state = game.state()

            if self._is_selfplay:
                probs = 0.8 * probs + 0.2 * np.random.dirichlet(0.3 * np.ones(len(probs)))

                move = acts[np.argmax(probs)]
                # move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(move, state)  # 更新根节点，并且复用子树
            else:
                move = acts[np.argmax(probs)]
                # move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1, state)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

# -*- coding: utf-8 -*-
import numpy as np
import copy

from constant import *


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
     定义蒙特卡洛树的节点
     每一个节点保存自己的Q值，先验概率P和访问计数N
    """

    # 初始化节点，包括父节点，子节点，N,Q，P，以及额外的奖励u的信息
    def __init__(self, parent, prior_p, state):
        self._parent = parent
        self._children = {}  # 子节点用字典表示，是一个从动作(落子)到节点的映射。
        self._n_visits = 0
        self._state = state
        self._Q = 0
        self._u = 0
        self._P = prior_p
        # self._game = game

    def expand(self, action_priors, game):
        duplicated_node = False
        parent_node = None
        parent_state = None

        for action, prob in action_priors:
            if action < 12:
                """
                Code for restrict dummy expand
                """
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

            # Add condition 'not duplicated_node'
            if not duplicated_node and action not in self._children:
                self._children[action] = TreeNode(self, prob, next_state)

    def select(self, c_puct):
        """
        选择，蒙特卡洛树的基本操作之一，根据论文的描述，选择当前最大的Q+u值的那个动作
        返回的是一个元组，包括选择的动作和孩子节点
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        更新，其实也就是backup，同样是蒙特卡洛树的基本操作之一，
        从叶节点更新自己的评价
        leaf_value是当前玩家视角下的叶节点价值
        """
        # 更新包括计数加一，Q值更新
        self._n_visits += 1
        # Q值采用滑动平均方法更新
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        递归更新所有祖先的相关信息，也就是递归调用update
        """
        # 如果不是根节点，就进行递归

        if self._parent:
            # print(leaf_value)
            self._parent.update_recursive(-leaf_value)

        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        获取价值，也就是论文中的Q+u，用来进行Select。
        c_puct是参数，是UCT算法的变体，实际上控制开发和探索的权衡
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """
        判断是否是叶子节点，没有孩子节点的就是叶子结点。
        """
        return self._children == {}

    # 判断是否是根节点，没有父节点的就是根节点
    def is_root(self):
        return self._parent is None

    def get_parent(self):
        return self._parent


class MCTS(object):
    """
    实现蒙特卡洛树搜索

    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1800):
        """
        初始化蒙特卡洛树搜索
        参数:
        policy_value_fn -- 是一个函数，输入棋盘状态，返回下一步的落子和概率，(action, probability)以及一个-1到1之间的分数。
        表示这一步导致的最后的胜负情况。
        c_puct -- 0到正无穷之间的一个数，越大意味着越依赖以前
        """
        self._root = TreeNode(None, 1.0, None)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    # Fix : get current_player param info when the first simulation started.
    def _playout(self, game, current_player):
        """
        单次的蒙特卡洛搜索的模拟，即从根节点到叶节点一次，获得叶子节点价值然后反向传导，更新所有祖先节点
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break
                # 贪婪选择下一步
            action, node = node.select(self._c_puct)
            game.step(action)  # 进行模拟落子一次
        # 评估叶子结点，得到一系列的 (action, probability)
        # 以及一个-1到1之间的价值
        # state = game.state()
        action_probs, leaf_value = self._policy(game)
        # 检查是否游戏结束
        end, winner = game.has_a_winner()
        # 没有结束，扩展节点，利用网络输出的先验概率
        if not end:
            # Add an incompleted code to make pawn avoid dead-end section.
            """
            if np.sum(game.actions()[:4]) <= 1:
                leaf_value = -1.0 if game.get_current_player == current_player else 1.0
            else:
            """
            node.expand(action_probs, game)
        # 结束了，返回真实的叶子结点值，不需要网络评估了。
        else:
            leaf_value = 1.0 if winner == game.get_current_player() else -1.0  # Fix bug that all winners are current player
            # print(leaf_value)
        # 迭代更新所有祖先节点
        # print("call update")
        node.update_recursive(leaf_value)

    def get_move_probs(self, game, temp=1e-3):
        """
        多次模拟，并且根据子节点访问的次数和温度系数计算下一步落子的概率。
        温度系数0-1之间，控制探索的权重，越靠近1，分布越均匀，多样性大，
        越越接近0，分布越尖锐，追求最强棋力
        """
        for n in range(self._n_playout):
            game_copy = copy.deepcopy(game)
            # state = game.state()
            # state_copy = copy.deepcopy(state)
            self._playout(game_copy, game_copy.get_current_player())
        # 根据访问次数计算落子概率
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        q_vals = [node._Q for act, node in self._root._children.items()]
        # print("-" * 30)
        # print("q_vals : ", q_vals)
        # print("-" * 30)
        return acts, act_probs

    def update_with_move(self, last_move, state):
        if last_move in self._root._children:  # 根据对面的落子，复用子树，
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:  # 否则，重新开始一个新的搜索树
            self._root = TreeNode(None, 1.0, state)

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
    def choose_action(self, game, temp=1e-3, return_prob=0):
        sensible_moves = game.actions()  # 获取所有可行的落子
        move_probs = np.zeros(12 + (BOARD_SIZE - 1) ** 2 * 2)  # 获取落子的概率，由神经网络输出
        if len(sensible_moves) > 0:  # 棋盘未耗尽时
            acts, probs = self.mcts.get_move_probs(game, temp)  # 获取落子以及对应的落子概率
            move_probs[list(acts)] = probs  # 将概率转到move_probs列表中
            state = game.state()

            if self._is_selfplay:

                probs = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))

                max_acts = np.argwhere(probs == np.amax(probs))[0]
                print(max_acts, acts)
                max_act = np.random.choice(max_acts)

                move = acts[max_act]

                # move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                self.mcts.update_with_move(move, state)  # 更新根节点，并且复用子树
            else:

                #move = np.random.choice(acts, p=probs)

                max_acts = np.argwhere(probs == np.amax(probs))[0]

                max_act = np.random.choice(max_acts)

                move = acts[max_act]


                self.mcts.update_with_move(-1, state)
            # location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

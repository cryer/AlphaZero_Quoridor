# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import time

import numpy as np
from collections import defaultdict, deque
from quoridor import Quoridor
from policy_value_net import PolicyValueNet

from mcts import MCTSPlayer

from torch.utils.tensorboard import SummaryWriter


from constant import BOARD_SIZE, WALL_NUM


writer = SummaryWriter()


class TrainPipeline(object):
    def __init__(self, init_model=None):
        # 棋盘参数
        self.game = Quoridor()
        # 训练参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 适应性调节学习速率
        self.temp = 1.0
        self.n_playout = 200
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 32  # 取1 测试ing
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 50
        self.kl_targ = 0.02
        self.check_freq = 5
        self.game_batch_num = 1000
        self.game_batch_num = 200
        
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000
        if init_model:
            self.policy_value_net = PolicyValueNet(model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet()
        # 设置电脑玩家信息
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playout, is_selfplay=1)

    # def get_equi_data(self, play_data):
    #     """
    #     数据集增强，获取旋转后的数据，因为五子棋也是对称的
    #     play_data: [(state, mcts_prob, winner_z), ..., ...]"""
    #     extend_data = []
    #     for state, mcts_porb, winner in play_data:
    #         equi_state = np.array([np.rot90(s,2) for s in state])
    #         equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(9, 9)), 2)
    #         extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
    #         # flip horizontally
    #         equi_state = np.array([np.fliplr(s) for s in equi_state])
    #         equi_mcts_prob = np.fliplr(equi_mcts_prob)
    #         extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
    #     return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集训练数据"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)  # 进行自博弈
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 数据增强
            # play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """训练策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)  # 获取mini-batch
        state_batch = [data[0] for data in mini_batch]  # 提取第一位的状态
        mcts_probs_batch = [data[1] for data in mini_batch]  # 提取第二位的概率
        winner_batch = [data[2] for data in mini_batch]  # 提取第三位的胜负情况
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)  # 输入网络计算旧的概率和胜负价值，这里为什么要计算旧的数据是因为需要计算
        #                                                                     新旧之间的KL散度来控制学习速率的退火
        # 开始训练epochs个轮次
        for i in range(self.epochs):
            valloss, polloss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                             self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)  # 计算新的概率和价值
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度发散的很不好，就提前结束训练
                break
        # 根据KL散度，适应性调节学习速率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        print(
            "kl:{:.5f},lr_multiplier:{:.3f},value loss:{},policy loss:[],entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, valloss, polloss, entropy, explained_var_old, explained_var_new))
        return valloss, polloss, entropy


    def run(self):
        try:
            self.collect_selfplay_data(50)
            count = 0
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)    # collect_s
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    valloss, polloss, entropy = self.policy_update()
                    print("VALUE LOSS: %0.3f " % valloss.item(), "POLICY LOSS: %0.3f " % polloss.item())
                    print("ENTROPY:",entropy)
                    # 保存loss

                    writer.add_scalar("Val Loss/train", valloss.item(), i)
                    writer.add_scalar("Policy Loss/train", polloss.item(), i)
                    writer.add_scalar("Entropy/train", entropy, i)

                if (i + 1) % self.check_freq == 0:
                    count += 1
                    print("current self-play batch: {}".format(i + 1))
                    # win_ratio = self.policy_evaluate()
                    # Add generation to filename
                    
                    self.policy_value_net.save_model('cp_gen_3_' + str(count) + '_' + str("%0.3f_" % valloss.item()) + str(time.strftime('%Y-%m-%d', time.localtime(time.time()))))  # 保存模型
        except KeyboardInterrupt:
            print('\n\rquit')


# Start
if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model=None)
    training_pipeline.run()

from __future__ import print_function

from quoridor import Quoridor
from policy_value_net import PolicyValueNet

from mcts import MCTSPlayer as A_Player
from pure_mcts import MCTSPlayer as B_Player


class TestTrainedAgent(object):
    def __init__(self, init_model=None, first_player=1):
        self.game = Quoridor()
        self.temp = 1.0
        self.c_puct = 5
        self.play_batch_size = 1
        self.alpha_playout = 100
        self.pure_playout = 100
        self.first = first_player

        self.alpha_player = A_Player(PolicyValueNet(model_file=init_model).policy_value_fn, c_puct=5,
                                     n_playout=self.alpha_playout, is_selfplay=0)
        self.pure_player = B_Player(c_puct=5, n_playout=self.pure_playout)  #
        self.alpha_win_total = 0
        self.alpha_win_first = 0
        self.alpha_draw_total = 0
        self.alpha_draw_first = 0

    def test_against_pure(self, n_games=1):
        for i in range(n_games):
            if self.first == 3:
                if i % 2 == 0:
                    winner = self.game.start_test_play(self.alpha_player, self.pure_player,
                                                       temp=self.temp, first=1)
                else:
                    winner = self.game.start_test_play(self.alpha_player, self.pure_player,
                                                       temp=self.temp, first=2)
            else:
                winner = self.game.start_test_play(self.alpha_player, self.pure_player,
                                                   temp=self.temp, first=self.first)

            if winner == 1:
                if i % 2 == 0:
                    self.alpha_win_first += 1
                    self.alpha_win_total += 1
                else:
                    self.alpha_win_total += 1
                print("{}th/{} game finished. Winner is alpha zero player".format((i + 1), n_games))
            elif winner == 0:
                if i % 2 == 0:
                    self.alpha_draw_first += 1
                    self.alpha_draw_total += 1
                else:
                    self.alpha_draw_total += 1
                print("{}th/{} game finished. Draw".format((i + 1), n_games))
            else:
                print("{}th/{} game finished. Winner is pure mcts player".format((i + 1), n_games))
            print("alpha zero win rate in first start : {:.2%}".format(self.alpha_win_first / (i + 1)))
            print("alpha zero win rate in second start : {:.2%}".format((self.alpha_win_total - self.alpha_win_first) / (i + 1)))
            print("alpha zero win rate total : {:.2%}".format(self.alpha_win_total / (i + 1)))
            print("alpha zero draw rate in first start : {:.2%}".format(self.alpha_draw_first / (i + 1)))
            print("alpha zero draw rate in second start : {:.2%}".format(
                (self.alpha_draw_total - self.alpha_draw_first) / (i + 1)))
            print("alpha zero draw rate total : {:.2%}".format(self.alpha_draw_total / (i + 1)))

    def run(self, epoch_num):
        try:
            self.test_against_pure(n_games=epoch_num)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    # init_model : alpha zero model file name
    # first_player : 1 - alpha zero, 2 - pure mcts, 3 - change first player when every game finish
    test_trained_agent = TestTrainedAgent(init_model=None, first_player=3)
    test_trained_agent.run(100)

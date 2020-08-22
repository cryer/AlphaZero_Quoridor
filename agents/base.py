import numpy as np


class BaseAgent:

    def choose_action(self, game):
        action = np.random.choice(game.actions())
        print("Choosing action {action}".format(action=action))
        return action


class RandomAgent:

    def choose_action(self, game):
        action = np.random.choice(game.actions())
        print("Choosing action {action}".format(action=action))
        return action

class RandomMoveAgent:
    def choose_action(self, game):
        legal_actions = game.actions()
        
        legal_move_actions = np.intersect1d(np.arange(12), legal_actions)

        return np.random.choice(legal_move_actions)


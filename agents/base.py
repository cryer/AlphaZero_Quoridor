import numpy as np

class BaseAgent:

    def choose_action(self, game):
        action = np.random.choice(game.actions())
        print("Choosing action {action}".format(action=action))
        return action

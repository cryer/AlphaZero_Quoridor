import numpy as np

class BaseAgent:

    def __init__(self, name, environment=None):
        self.name = name
        self.environment = environment

    def choose_action(self):
        action = np.random.choice(self.environment.valid_actions)
        pawn_actions = [a for a in self.environment.valid_actions if a < 12]
        action = action = np.random.choice(pawn_actions)
        print("Choosing action {action}".format(action=action))
        return action

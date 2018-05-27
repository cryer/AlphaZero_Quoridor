from .base import BaseAgent


class ManualCLIAgent(BaseAgent):
    def choose_action(self):
        print("Current Board")
        self.environment.print_board()
        print("Available Actions")
        valid = self.environment.actions
        print(valid)
        action = int(input("Choose Action: "))
        while action not in valid:
            print("Invalid Action: {action} - please select a valid action".format(action=action))
            print(valid)
            action = int(input("Choose Action: "))
        return action


class ManualPygameAgent(BaseAgent):
    def __init__(self, name):
        self.name = name
        self._action = None

    def receive_action(self, action):
        self._action = action

    def choose_action(self):
        return self._action


class HistoricalPygameAgent(BaseAgent):
    def __init__(self, name, moveset):
        self.name = name
        self.moveset = moveset

    def choose_action(self):
        for move in self.moveset:
            yield move

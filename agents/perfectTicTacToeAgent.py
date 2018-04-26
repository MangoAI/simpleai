from agents.agent import Agent
from games.ninarowInterface import NInARow, Move
import numpy as np

class PerfectTicTacToeAgent(Agent):

    def __init__(self):
        self.valuesAndActions = NInARow.loadValuesAndActions()

    def getMove(self, board):
        actions = self.valuesAndActions[board]['actions']
        action = actions[np.random.randint(0, len(actions))]
        return Move(board.getTurn(), action[0], action[1])
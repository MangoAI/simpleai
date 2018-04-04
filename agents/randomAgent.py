from agent import Agent
import numpy as np

class RandomAgent(Agent):

    def __init__(self):
        pass

    def getMove(self, board):
        return np.random.choice(list(board.getLegalMoves()))
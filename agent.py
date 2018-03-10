from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def getMove(self, board):
        pass

class RandomAgent(Agent):

    def __init__(self):
        pass

    def getMove(self, board):
        return np.random.choice(list(board.getLegalMoves()))
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def getMove(self, board):
        pass

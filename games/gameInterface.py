from abc import ABC, abstractmethod

ONGOING = "GAME STILL ONGOING"
DRAW = "DRAW"

class GameInterface(ABC):

    @abstractmethod
    def getTurn(self):
        pass

    @abstractmethod
    def getTurnOrder(self):
        pass

    @abstractmethod
    def getLegalMoves(self):
        pass

    @abstractmethod
    def play(self, move):
        pass

    @abstractmethod
    def getResult(self):
        """
        Returns winning player, DRAW, or ONGOING
        :return:
        """
        pass

    @abstractmethod
    def getHistory(self):
        """
        Return a tuple of Moves taken in game
        :return:
        """
        pass

    @abstractmethod
    def undo(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass
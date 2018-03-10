from abc import ABC, abstractmethod
from ninarow_game.board import Board as NInARowGameBoard


class BoardInterface(ABC):
    DRAW = 0
    ONGOING = 2

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
    def undo(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

class NInARowBoard(BoardInterface):

    def __init__(self, board_dim, n):
        self.board = NInARowGameBoard(board_dim, n)

    def getLegalMoves(self):
        return self.board.getLegalMoves()

    def play(self, move):
        return self.board.play(move)

    def getResult(self):
        return self.board.getResult()

    def undo(self):
        return self.board.undo()

    def copy(self):
        newBoard = NInARowBoard(1,1)
        newBoard.board = self.board.copy()
        return newBoard

    def __repr__(self):
        return str(self.board.board)
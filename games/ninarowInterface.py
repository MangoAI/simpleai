from .gameInterface import GameInterface, ONGOING
from .ninarow.board import Board, Move
from .ninarow import ninarow
import numpy as np

class NInARow(GameInterface):

    turns = (ninarow.WHITE, ninarow.BLACK)
    WHITE = ninarow.WHITE
    BLACK = ninarow.BLACK

    def __init__(self, board_dim, n):
        self.board = Board(board_dim, n)

    def getTurn(self):
        return self.board.current_player

    def getPrevTurn(self):
        return ninarow.WHITE if self.getTurn() is ninarow.BLACK else ninarow.WHITE

    def getNextTurn(self):
        return ninarow.WHITE if self.getTurn() is ninarow.BLACK else ninarow.WHITE

    def getLegalMoves(self):
        return self.board.getLegalMoves()

    def play(self, move):
        """
        Can take a move, or can take a tuple (x, y)
        :param move:
        :return:
        """
        if type(move) != Move:
            move = Move(self.getTurn(), move[0], move[1])
        return self.board.play(move)

    def getResult(self):
        result = self.board.getResult()
        return ONGOING if result is ninarow.ONGOING else result

    def getHistory(self):
        return tuple(self.board.history)

    def undo(self):
        return self.board.undo()

    def copy(self):
        newBoard = NInARow(1, 1)
        newBoard.board = self.board.copy()
        return newBoard

    def featurize(self):
        whites = self.board.board == ninarow.WHITE
        blacks = self.board.board == ninarow.BLACK
        return np.append([int(self.getTurn() == ninarow.BLACK)], np.append(whites, blacks))

    def __repr__(self):
        return str(self.board.board)

    def __hash__(self):
        return (self.board.current_player, str(self.board.board)).__hash__()
from .gameInterface import GameInterface, ONGOING
from .ninarow.board import Board
from .ninarow import ninarow

class NInARow(GameInterface):

    turns = (ninarow.WHITE, ninarow.BLACK)

    def __init__(self, board_dim, n):
        self.board = Board(board_dim, n)

    def getTurn(self):
        return self.board.current_player

    def getLegalMoves(self):
        return self.board.getLegalMoves()

    def play(self, move):
        return self.board.play(move)

    def getResult(self):
        result = self.board.getResult()
        return ONGOING if result is ninarow.ONGOING else result

    def undo(self):
        return self.board.undo()

    def copy(self):
        newBoard = NInARow(1, 1)
        newBoard.board = self.board.copy()
        return newBoard

    def __repr__(self):
        return str(self.board.board)

    def __hash__(self):
        return (self.board.current_player, str(self.board.board)).__hash__()
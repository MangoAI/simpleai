from abc import ABC, abstractmethod
from ninarow_game.board import Board as NInARowGameBoard
from ninarow_game import ninarow
import chess

WIN_SCORE = 999999999
LOSE_SCORE = -999999999
DRAW_SCORE = 0
ONGOING = "GAME STILL ONGOING"
DRAW = "DRAW"

class BoardInterface(ABC):

    @abstractmethod
    def getTurn(self):
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

class NInARowBoard(BoardInterface):

    def __init__(self, board_dim, n):
        self.board = NInARowGameBoard(board_dim, n)

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
        newBoard = NInARowBoard(1,1)
        newBoard.board = self.board.copy()
        return newBoard

    def __repr__(self):
        return str(self.board.board)

    def __hash__(self):
        return (self.board.current_player, self.board.board).__hash__()

class ChessBoard(BoardInterface):

    def __init__(self):
        self.board = chess.Board()

    def getTurn(self):
        return self.board.turn

    def getLegalMoves(self):
        return self.board.legal_moves

    def play(self, move):
        return self.board.push(move)

    def getResult(self):
        if not self.board.is_game_over():
            return ONGOING
        if self.board.is_checkmate():
            return chess.WHITE if self.board.turn is chess.BLACK else chess.WHITE
        return DRAW

    def undo(self):
        return self.board.pop()

    def copy(self):
        newBoard = ChessBoard()
        newBoard.board = self.board.copy()
        return newBoard

    def __repr__(self):
        return str(self.board)

    def __hash__(self):
        return repr(self.board).__hash__()

    def getNumPiece(self, piece, player):
        return len(list(self.board.pieces(piece, player)))
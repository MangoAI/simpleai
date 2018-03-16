import chess
from .gameInterface import GameInterface, DRAW, ONGOING

class Chess(GameInterface):

    turns = (chess.WHITE, chess.BLACK)

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
        newBoard = Chess()
        newBoard.board = self.board.copy()
        return newBoard

    def __repr__(self):
        return str(self.board)

    def __hash__(self):
        return repr(self.board).__hash__()

    def getNumPiece(self, piece, player):
        return len(list(self.board.pieces(piece, player)))
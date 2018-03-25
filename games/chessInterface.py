import chess
from .gameInterface import GameInterface, DRAW, ONGOING

class Chess(GameInterface):

    turns = (chess.WHITE, chess.BLACK)

    def __init__(self):
        self.board = chess.Board()
        self.history = []

    def getTurn(self):
        return self.board.turn

    def getPrevTurn(self):
        return chess.WHITE if self.getTurn() is chess.BLACK else chess.WHITE

    def getNextTurn(self):
        return chess.WHITE if self.getTurn() is chess.BLACK else chess.WHITE

    def getLegalMoves(self):
        return self.board.legal_moves

    def play(self, move):
        self.history.append(move)
        return self.board.push(move)

    def getResult(self):
        if not self.board.is_game_over():
            return ONGOING
        if self.board.is_checkmate():
            return chess.WHITE if self.board.turn is chess.BLACK else chess.WHITE
        return DRAW

    def getHistory(self):
        return tuple(self.history)

    def undo(self):
        self.history.pop()
        return self.board.pop()

    def copy(self):
        newBoard = Chess()
        newBoard.board = self.board.copy()
        return newBoard

    def __repr__(self):
        return str(self.board)

    def __hash__(self):
        return (self.getTurn(), repr(self.board)).__hash__()

    def getNumPiece(self, piece, player):
        return len(list(self.board.pieces(piece, player)))
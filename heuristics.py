import chess
import numpy as np

def chessPieceScore(board, turn):
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    weights = np.array([1, 3, 3, 5, 9])
    score = weights.dot([board.getNumPiece(piece, turn) for piece in pieces])
    oppScore = weights.dot([board.getNumPiece(piece, 1-turn) for piece in pieces])
    return score - oppScore
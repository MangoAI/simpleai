import chess
import numpy as np
import sys
sys.path.insert(0, '..')
from games import gameInterface

WINSCORE = 999999999999
LOSESCORE = -999999999999

def chessPieceScore(board, turn):
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    weights = np.array([1, 3, 3, 5, 9])
    score = weights.dot([board.getNumPiece(piece, turn) for piece in pieces])
    oppScore = weights.dot([board.getNumPiece(piece, 1-turn) for piece in pieces])
    return score - oppScore

def ninarowWinScore(board, turn):
    result = board.getResult()
    if result == gameInterface.ONGOING:
        return 0
    elif result == turn:
        return WINSCORE
    return LOSESCORE

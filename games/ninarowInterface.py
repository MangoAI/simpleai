from .gameInterface import GameInterface, ONGOING, DRAW
from .ninarow import Board, Move
from .ninarow import WHITE as nWHITE, BLACK as nBLACK, ONGOING as nONGOING, DRAW as nDRAW
import numpy as np
import pickle

class NInARow(GameInterface):

    WHITE = nWHITE
    BLACK = nBLACK
    DRAW = DRAW
    turns = (WHITE, BLACK)
    ONGOING = ONGOING


    def __init__(self, board_dim, n):
        self.board = Board(board_dim, n)

    def getTurn(self):
        return self.board.current_player

    def getTurnOrder(self):
        return self.turns

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
        self.board.play(move)
        return self

    def getResult(self):
        result = self.board.getResult()
        if result is nONGOING:
            return ONGOING
        if result is nDRAW:
            return DRAW
        return result

    def getHistory(self):
        return tuple(self.board.history)

    def undo(self):
        return self.board.undo(1)

    def copy(self):
        newBoard = NInARow(1, 1)
        newBoard.board = self.board.copy()
        return newBoard

    def featurize(self):
        whites = self.board.board == NInARow.WHITE
        blacks = self.board.board == NInARow.BLACK
        return np.append([int(self.getTurn() == NInARow.BLACK)], np.append(whites, blacks))

    def __repr__(self):
        return str(self.board.board)

    def __hash__(self):
        return (self.board.current_player, str(self.board.board)).__hash__()

    def __eq__(self, other):
        return self.board.current_player == other.board.current_player \
            and np.array_equal(self.board.board, other.board.board)

    @staticmethod
    def loadScores():
        with open('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/pickles/tictactoeScores.pickle', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def loadValuesAndActions():
        with open('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/pickles/tictactoeValuesAndActions.pickle',
                  'rb') as f:
            return pickle.load(f)

    @staticmethod
    def scorer(scores, board):
        if board in scores:
            return scores[board]
        result = board.getResult()
        if result is not ONGOING:
            if result is DRAW:
                scores[board] = 0
            else:
                scores[board] = result
            return scores[board]
        newScores = []
        for move in board.getLegalMoves():
            newBoard = board.copy()
            newBoard.play(move)
            newScores.append(NInARow.scorer(scores, newBoard))
        scores[board] = max(newScores) if board.getTurn() is NInARow.WHITE else min(newScores)
        return scores[board]

    @staticmethod
    def bestActioner():
        valuesAndActions = {}
        scores = NInARow.loadScores()
        for board, score in scores.items():
            value = score
            legalMoves = board.getLegalMoves()
            if legalMoves:
                moveScores = {}
                for move in legalMoves:
                    newBoard = board.copy()
                    newBoard.play(move)
                    moveScores[(move.x, move.y)] = scores[newBoard]
                bestScore = max(moveScores.values()) if board.getTurn() is NInARow.WHITE else min(moveScores.values())
                moves = [move for move in moveScores if moveScores[move] == bestScore]
                valuesAndActions[board] = {'value': score, 'actions': moves}
            else:
                valuesAndActions[board] = {'value': score, 'actions': []}
        return valuesAndActions

    @staticmethod
    def createPickles():
        scores = {}
        board = NInARow(3, 3)
        NInARow.scorer(scores, board)
        with open('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/pickles/tictactoeScores.pickle', 'wb') as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)
        valuesAndActions = NInARow.bestActioner()
        with open('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/pickles/tictactoeValuesAndActions.pickle', 'wb') as f:
            pickle.dump(valuesAndActions, f, pickle.HIGHEST_PROTOCOL)

#
if __name__ == '__main__':
    pass
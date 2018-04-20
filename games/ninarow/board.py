import numpy as np
from .ninarow import WHITE, ONGOING, getOtherPlayer, DRAW

class Move:

    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def getCoords(self):
        return np.array((self.x, self.y))

    def __repr__(self):
        return str((self.x, self.y))


class Board:

    def __init__(self, board_dim, n):
        self.board_dim = board_dim
        self.n = n
        self.board = np.zeros((board_dim, board_dim))
        self.current_player = WHITE
        self.history = [] # list of moves

    def getLegalMoves(self):
        if self.getResult() is ONGOING:
            inds = np.where(self.board == 0)
            return [Move(self.current_player, inds[0][i], inds[1][i]) for i in range(len(inds[0]))]
        else:
            return []

    def play(self, move):
        x = move.x
        y = move.y
        assert self.board[x][y] == 0
        self.history.append(move)
        self.board[x][y] = move.player
        self.current_player = getOtherPlayer(move.player)
        return self

    def getPiece(self, x, y):
        return self.board[x][y]

    def getResult(self):
        """
        If game is over, returns WHITE, BLACK, or DRAW
        Else, returns ONGOING
        :return:
        """
        if len(self.history) < 2*self.n - 1:
            return ONGOING

        lastMove = self.history[-1]
        lastPiece, lastCoord = lastMove.player, lastMove.getCoords()
        directions = np.array([(0, 1), (1, 0), (1, 1), (1, -1)])
        for direction in directions:
            count = 1
            winners = [lastCoord]
            for sign in (-1, 1):
                for i in range(1, 5):
                    pos = lastCoord + sign*i*direction
                    x, y = pos[0], pos[1]
                    if not ((0 <= x < self.board_dim) and (0 <= y < self.board_dim)):
                        break
                    if self.board[x][y] == lastPiece:
                        count += 1
                        winners.append((x,y))
                        if count == self.n:
                            return lastPiece
                    else:
                        break
        if len(self.history) == self.board_dim * self.board_dim:
            return DRAW
        return ONGOING

    def undo(self, m):
        """
        Undoes the last m moves.
        :param m:
        :return:
        """
        if len(self.history) < m:
            return Board(self.board_dim, self.n)
        toUndo = self.history[-m:]
        self.history = self.history[:-m]
        for move in toUndo:
            self.board[move.x][move.y] = 0

    def copy(self):
        newBoard = Board(self.board_dim, self.n)
        newBoard.board = np.array(self.board)
        newBoard.history = self.history[:]
        newBoard.current_player = self.current_player
        return newBoard

    def __repr__(self):
        return str(self.board)

if __name__ == "__main__":
    pass
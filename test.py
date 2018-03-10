import chess
from abc import ABC, abstractmethod
import numpy as np

def getNumPiece(board, piece, player):
    return len(list(board.pieces(piece, player)))

def getBoardPieceScore(board, turn):
    """
    :param board:
    :param turn: chess.WHITE or chess.BLACK
    :return:
    """


    numPawns = getNumPiece(board, chess.PAWN, turn)
    numKnights = getNumPiece(board, chess.KNIGHT, turn)
    numBishops = getNumPiece(board, chess.BISHOP, turn)
    numRooks = getNumPiece(board, chess.ROOK, turn)
    numQueens = getNumPiece(board, chess.QUEEN, turn)
    numKings = getNumPiece(board, chess.KING, turn)

    oppNumPawns = getNumPiece(board, chess.PAWN, 1-turn)
    oppNumKnights = getNumPiece(board, chess.KNIGHT, 1-turn)
    oppNumBishops = getNumPiece(board, chess.BISHOP, 1-turn)
    oppNumRooks = getNumPiece(board, chess.ROOK, 1-turn)
    oppNumQueens = getNumPiece(board, chess.QUEEN, 1-turn)
    oppNumKings = getNumPiece(board, chess.KING, 1-turn)

    score = numPawns + 3*numKnights + 3*numBishops + 5*numRooks + 9*numQueens + 10000*numKings
    oppScore = oppNumPawns + 3*oppNumKnights + 3*oppNumBishops + 5*oppNumRooks + 9*oppNumQueens + 10000*oppNumKings
    return score - oppScore + (np.random.rand()/10000)


class Game:

    def __init__(self, player1, player2):
        self.board = chess.Board()
        self.players = [player1, player2]

    def play(self):
        turns = 0
        currentPlayer = chess.WHITE
        while not self.board.is_game_over():
            print(turns)
            self.board.push(self.players[currentPlayer].getMove(self.board))
            currentPlayer = not currentPlayer
            turns += 1

            if turns%10 == 0:
                print(turns)
                print(self.board)
        print(turns)
        print(self.board)

class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def getMove(self, board):
        pass

class RandomAgent(Agent):

    def __init__(self):
        pass

    def getMove(self, board):
        return np.random.choice(list(board.legal_moves))

class MinMaxAgent(Agent):
    def __init__(self, maxDepth):
        self.maxDepth = maxDepth

    def getMove(self, board):
        board, move, score = self.minMax(board, self.maxDepth)
        return move

    def minMax(self, board, depth):
        """
        :param board:
        :param depth:
        :return: (board, move, score)
        """
        if board.is_game_over():
            if board.is_checkmate():
                return board, None, 10000
            else:
                return board, None, -3
        if depth == 0:
            return board, None, getBoardPieceScore(board, board.turn)
        else:
            newDepth = depth - 1
            boards = [(board.copy(), move) for move in board.legal_moves]
            [board.push(move) for board, move in boards]
            maxBoards = []
            for newBoard, move in boards:
                if newBoard.is_game_over():
                    if board.is_checkmate():
                        return newBoard, move, 10000
                    else:
                        maxBoards.append(( newBoard, None, -3 ))
                else:
                    minBoards = [(newBoard.copy(), newMove) for newMove in newBoard.legal_moves]
                    [minBoard.push(newMove) for minBoard, newMove in minBoards]
                    minBoards = [minBoard for minBoard, newMove in minBoards]
                    minBoard, minMove, minScore = min([self.minMax(minBoard, newDepth) for minBoard in minBoards],
                             key=lambda x: x[2])
                    maxBoards.append((minBoard, move, minScore))
            winningBoard, winningMove, winningScore = max(maxBoards, key=lambda x: x[2])
            return winningBoard, winningMove, winningScore

# class MemoryAgent(Agent):
#     def __init__(self):
#         self.memory = {}
#
#     @abstractmethod
#     def getMove(self, board):
#         pass
#
#     def getBoardScore(self, board):
#         if board in self.memory:
#             return self.memory[board]
#         return getBoardPieceScore(board, board.turn)
#
#
# class BoardStateNode:
#     def __init__(self, agent: MemoryAgent, parent, board):
#         self.agent = agent
#         self.parent = parent
#         self.board = board
#         self.score = self.agent.getBoardScore(self.board)
#         self.children = []
#
#     def getChildrenNodes(self):
#         if len(self.children) == len(self.board.legal_moves):
#             return self.children
#         self.children = []
#         for move in self.board.legal_moves:
#             board = self.board.copy()
#             board.push(move)
#             node = BoardStateNode(self.agent, self, board)
#             self.children.append(node)


#
# class MemoABAgent(Agent):
#     def __init__(self, maxDepth):
#         self.maxDepth = maxDepth
#
#     def getMove(self, board):
#         self.memory = {}
#         # return move
#
#     def memoAB(self, board, depth):
#
#
#
#     def getChildrenNodes(self,):

if __name__ == '__main__':
    game = Game(RandomAgent(), MinMaxAgent(1))
    game.play()
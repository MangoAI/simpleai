from agent import RandomAgent, MinMaxAgent
from boardInterface import ChessBoard, ONGOING
from ninarow_game.ninarow import BLACK, WHITE
from heuristics import chessPieceScore
import chess

class Game:

    def __init__(self, board, agent1, agent2):
        self.board = board
        self.agents = [agent1, agent2]

    def play(self):
        currentPlayer = 0
        turns = 0
        while self.board.getResult() == ONGOING:
            self.board.play(self.agents[currentPlayer].getMove(self.board))
            currentPlayer = (currentPlayer + 1)%len(self.agents)
            turns += 1
            print(turns)
            if turns%10 == 0:
                print(turns, chessPieceScore(self.board, chess.BLACK))
                print(self.board)
            # if turns == 100:
            #     self.agents[1].maxDepth = 3
        print(turns)
        print(self.board)
        print(self.board.getResult())

if __name__ == '__main__':
    g = Game(ChessBoard(), RandomAgent(), MinMaxAgent([chess.BLACK, chess.WHITE], chessPieceScore, 1, 4))
    # g = Game(ChessBoard(), MinMaxAgent([WHITE, BLACK], chessPieceScore, 1, 4),  RandomAgent())

    g.play()
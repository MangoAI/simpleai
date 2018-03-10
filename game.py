from agent import Agent, RandomAgent
from boardInterface import BoardInterface, NInARowGameBoard

class Game:

    def __init__(self, board, agent1, agent2):
        self.board = board
        self.agents = [agent1, agent2]

    def play(self):
        currentPlayer = 0
        turns = 0
        while self.board.getResult() == BoardInterface.ONGOING:
            self.board.play(self.agents[currentPlayer].getMove(self.board))
            currentPlayer = not currentPlayer
            turns += 1

            if turns%10 == 0:
                print(turns)
                print(self.board)
        print(turns)
        print(self.board)
        print(self.board.getResult())

if __name__ == '__main__':
    g = Game(NInARowGameBoard(10, 5), RandomAgent(), RandomAgent())
    g.play()
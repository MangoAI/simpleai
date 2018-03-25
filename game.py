from agents import RandomAgent, MinMaxAgent, LearnerAgent, AgentTrainer
from games import Chess, ONGOING, NInARow
from games.ninarow.board import Move
import numpy as np

class Game:

    def __init__(self, board, agent1, agent2):
        self.board = board
        self.agents = [agent1, agent2]

    def play(self):
        currentPlayer = 0
        while self.board.getResult() == ONGOING:
            self.board.play(self.agents[currentPlayer].getMove(self.board))
            currentPlayer = (currentPlayer + 1)%len(self.agents)
            # print(self.board)
        return self.board.getResult()

def manyGames():
    wins = 0
    for i in range(1, 101):
        game = Game(NInARow(3, 3), MinMaxAgent(NInARow.turns, lambda a, b: 0, 5, 100), RandomAgent())
        game.board.play(Move(1, 0, 2))
        game.board.play(Move(-1, 1, 0))
        result = game.play()
        if result == NInARow.turns[0]:
            wins += 1
        print(i, wins / i)

def randomModel(board, player):
    n = len(board.getLegalMoves())
    # return 0, np.ones(n)/n
    return np.random.rand()/10000, np.ones(n)/n

if __name__ == '__main__':
    AgentTrainer('agents/data', randomModel).train()
    # wins = 0
    # for i in range(1, 1000):
    #     learner1 = LearnerAgent(randomModel, np.sqrt(2), 50)
    #     learner2 = LearnerAgent(randomModel, np.sqrt(2), 50)
    #     random = RandomAgent()
    #     game = Game(NInARow(3,3), learner1, random)
    #     # game = Game(NInARow(3,3), learner1, learner2)
    #     # game = Game(NInARow(3,3), random, learner2)
    #     # game = Game(NInARow(3, 3), random, random)
    #     result = game.play()
    #     if result == NInARow.turns[0]:
    #         wins += 1
    #     print(i, wins / i)
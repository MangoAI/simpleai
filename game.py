from agents import RandomAgent, LearnerAgent, NInARowTrainer
from agents.learnerAgent import compareModels
from agents.perfectTicTacToeAgent import PerfectTicTacToeAgent
from games import ONGOING, NInARow
from agents.models import TicTacToeModel
from agents.models.model import testTicTacToeModel, NIARKerasFeedForwardModel, RandomModel, PerfectTicTacToe
from data.ninarowData import NInARowData
import numpy as np
import os
import pandas as pd

class Game:

    def __init__(self, board, agent1, agent2):
        self.board = board
        self.agents = [agent1, agent2]

    def play(self):
        currentPlayer = 0
        while self.board.getResult() == ONGOING:
            self.board.play(self.agents[currentPlayer].getMove(self.board))
            currentPlayer = (currentPlayer + 1)%len(self.agents)
        return self.board.getResult()

def testLastModel(trainer):
    modelFilename = trainer.data['modelFiles'][-1]
    model = NIARKerasFeedForwardModel(3, 3, [256, 256])
    model.load(modelFilename)
    testResults = testTicTacToeModel(model)
    totalBoards = testResults['totalBoards']
    avgValueDistance = testResults['avgValueDistance']
    correctActionPercentage = testResults['correctActionPercentage']
    avgConfidenceDistance = testResults['avgConfidenceDistance']
    correctActions = testResults['correctActions']
    actionsTaken = testResults['actionsTaken']

    with open(os.path.join(trainer.directory, 'train_results.txt'), "a") as f:
        f.write("Model {0}\n".format(len(trainer.data['modelFiles'])))
        f.write("Total number of boards evaluated: {0}\n".format(totalBoards))
        f.write("Average value distance: {0}\n".format(np.round(avgValueDistance, 4)))
        f.write("Correct actions: {0} out of {1} ({2}%)\n".format(
            correctActions, actionsTaken,
            np.round(correctActionPercentage, 4)))
        f.write("Average confidence distance: {0}\n".format(np.round(avgConfidenceDistance, 4)))
        f.write("\n")

if __name__ == '__main__':
    # hi = TicTacToeModel([256,256]).train()

    model = NIARKerasFeedForwardModel(3, 3, [256, 256])
    model.initialize()
    trainer = NInARowTrainer("data/ninarow/tictactoe/master22", model, 3, 3,
                                                      curiosity=np.sqrt(2),
                                                      max_depth=25,
                                                      trainEpochs=1000,
                                                      stochasticExploration=True, stochasticDecision=False)
    # trainer.trainModel(4000)
    trainer.train(100, 1000,
                  learnFromPastRounds=5,
                  agent1=PerfectTicTacToeAgent(),
                  agent2=None,
                  learnFromAgent1=True,
                  learnFromAgent2=True)
    #
    # results = compareModels(lambda: NInARow(3, 3), trainer.loadModel(-1), RandomModel(), 100, np.sqrt(2), 25)

    # model = NIARKerasFeedForwardModel(3, 3, [256, 256])
    # trainer = NInARowTrainer("data/ninarow/tictactoe19", model, 3, 3,
    #                                                   curiosity=np.sqrt(2),
    #                                                   max_depth=25,
    #                                                   trainEpochs=1000,
    #                                                   stochasticExploration=True, stochasticDecision=False)
    # trainer.train(10, 1000)
    # pModel = PerfectTicTacToe()
    # model1 = trainer.loadModel(-1)
    # model2 = RandomModel()
    # results = compareModels(lambda: NInARow(3,3), PerfectTicTacToe(), RandomModel(), 100, np.sqrt(2), 25)
    # data = NInARowData("/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/tictactoe16/training_data.pickle")
    # wins, draws, losses = 0, 0, 0
    # for game in data.games:
    #     result = game.getWinner()
    #     if result == 0:
    #         draws += 1
    #     elif result == 1:
    #         wins += 1
    #     elif result == 2:
    #         losses += 1
    # total = wins + losses + draws
    # print("{0} wins out of {1}: {2}%".format( wins, total, np.round(wins / total, 4)*100) )
    # print("{0} losses out of {1}: {2}%".format( losses, total, np.round(losses / total, 4) * 100))
    # print("{0} draws out of {1}: {2}%".format( draws, total, np.round(draws / total, 4) * 100))


    # board = NInARow(3, 3)
    # model = NIARKerasFeedForwardModel(3, 3, [256, 256])
    # model.initialize()
    #
    # trainer = NInARowTrainer("data/ninarow/tictactoe19", model, 3, 3,
    #                          curiosity=np.sqrt(2),
    #                          max_depth=25,
    #                          trainEpochs=1000,
    #                          stochasticExploration=True, stochasticDecision=False)
    #
    # testLastModel(trainer)
    # for i in range(3):
    #     print("Training iteration {0}".format(i + 1))
    #     trainer.train(1, 1000)
    #     testLastModel(trainer)
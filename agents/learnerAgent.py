from .agent import Agent
from .models.model import NInARowData, TicTacToeModel, NIARKerasFeedForwardModel, testTicTacToeModel, RandomModel
import numpy as np
import sys
sys.path.insert(0, '..')
from games import gameInterface, NInARow
import os
import datetime
import json
import pickle
from tqdm import tqdm

def weightedSample(arr, p):
    p -= np.min(p)
    if np.sum(p) > 0:
        p = p/np.sum(p)
    else:
        p = np.ones(len(p))/len(p)
    return np.random.choice(arr, p=p)

class State:

    def __init__(self, board):
        self.board = board

class RolloutMCTSTree:

    def __init__(self, rootBoard, curiosity):
        self.root = RolloutNode(self, rootBoard)
        self.turn = rootBoard.getTurn()
        self.curiosity = curiosity

    def select(self, node):
        if node.visits == 0 or not node.getChildren():  # Has never been visited before or leaf
            node.visits += 1
            return node
        if self.stochasticExploration:
            child = weightedSample(node.getChildren(), node.getUCTs(self.curiosity))
        else:
            child = node.getChildren()[np.argmax(node.getUCTs(self.curiosity))]
        deepestNode = self.select(child)
        for turn in node.board.getTurnOrder():
            node.values[turn] = (node.visits * node.getValue(turn) + deepestNode.getValue(turn)) / (node.visits + 1)

        node.visits += 1
        return deepestNode

    def getBestAction(self):
        return max(self.root.getChildren(), key=lambda c: c.value).board.getHistory()[-1]

class RolloutNode:

    def __init__(self, tree, board):
        self.tree = tree
        self.board = board
        self.value = 0
        self.children = None

    def getChildren(self):
        if not self.children:
            self.children = []
            for move in self.board.getLegalMoves():
                board = self.board.copy()
                board.play(move)
                if board not in self.tree.boardToNode:
                    self.tree.boardToNode[board] = Node(self.tree, board)
                self.children.append(self.tree.boardToNode[board])
        return self.children


class MCTSTree:

    def __init__(self, rootBoard, model, curiosity, stochasticExploration, stochasticDecision):
        """
        :param model: function that takes a State and player (turn) and returns a tuple (value, policyDistribution)
        """
        self.root = Node(self, rootBoard)
        self.turn = rootBoard.getTurn()
        self.model = model
        self.curiosity = curiosity
        self.stochasticExploration = stochasticExploration
        self.stochasticDecision = stochasticDecision
        self.boardToNode = {self.root.board: self.root}
        self.root.getChildren() # initializes children

    def select(self, node):
        if node.visits == 0 or not node.getChildren(): # Has never been visited before or leaf
            node.visits += 1
            return node
        if self.stochasticExploration:
            child = weightedSample(node.getChildren(), node.getUCTs(self.curiosity))
        else:
            child = node.getChildren()[np.argmax(node.getUCTs(self.curiosity))]
        deepestNode = self.select(child)
        for turn in node.board.getTurnOrder():
            node.values[turn] = (node.visits * node.getValue(turn) + deepestNode.getValue(turn)) / (node.visits + 1)

        node.visits += 1
        return deepestNode

    def getBestAction(self):
        if self.stochasticDecision:
            nextNode = weightedSample(self.root.getChildren(), [c.getValue(self.turn) for c in self.root.getChildren()])
        else:
            nextNode = max(self.root.getChildren(), key=lambda c: c.getValue(self.turn))
        return nextNode.board.getHistory()[-1]

class Node:

    def __init__(self, tree, board):
        self.tree = tree
        self.board = board
        self.values = {}
        self.visits = 0

        self.boardHash = board.__hash__()
        self.children = None
        self.policy = None

    def getUCTs(self, curiosity):
        turn = self.board.getTurn()
        values = np.array([child.getValue(turn) for child in self.getChildren()], dtype='float')
        # Renormalize values between 0 and 1
        values -= np.min(values)
        if (np.sum(values) > 0):
            values /= np.sum(values)
        visits = np.array([child.visits for child in self.getChildren()])
        return values + curiosity * self.getPolicy() * np.sqrt(np.sum(visits)) / (1 + visits)

    def getValue(self, player): # Returns value of node from perspective of player
        if player not in self.values:
            result = self.board.getResult()
            if result is gameInterface.ONGOING:
                value, policy = self.tree.model.getValueAndPolicy(self.board, player)
                self.values[player] = value
            else:
                if result == gameInterface.DRAW:
                    self.values = {turn: 0 for turn in self.board.turns}
                else:
                    self.values = {turn: 1 if turn == result else -1 for turn in self.board.getTurnOrder()}
        return self.values[player]

    def getPolicy(self):
        if self.policy is None:
            self.policy = self.tree.model.getValueAndPolicy(self.board, self.board.getTurn())
        return self.policy[1]

    def getChildren(self):
        if not self.children:
            self.children = []
            for move in self.board.getLegalMoves():
                board = self.board.copy()
                board.play(move)
                if board not in self.tree.boardToNode:
                    self.tree.boardToNode[board] = Node(self.tree, board)
                self.children.append(self.tree.boardToNode[board])
        return self.children

class LearnerAgent(Agent):

    def __init__(self, model, curiosity, maxNodes, stochasticExploration, stochasticDecision):
        self.model = model
        self.curiosity = curiosity
        self.maxNodes = maxNodes
        self.stochasticExploration = stochasticExploration
        self.stochasticDecision = stochasticDecision
        self.tree = None

    def getMove(self, board):
        self.tree = MCTSTree(board, self.model, self.curiosity, self.stochasticExploration, self.stochasticDecision)
        for i in range(self.maxNodes):
            self.tree.select(self.tree.root)
        return self.tree.getBestAction()

class NInARowTrainer:

    def __init__(self, directory, model,
                 board_dim, n,
                 curiosity, max_depth,
                 stochasticExploration, stochasticDecision,
                 trainEpochs=100):
        self.startTime = datetime.datetime.now()
        self.directory = os.path.abspath(directory)
        self.board_dim = board_dim
        self.n = n
        self.curiosity = curiosity
        self.max_depth = max_depth
        self.model = model
        self.stochasticExploration = stochasticExploration
        self.stochasticDecision = stochasticDecision
        self.trainEpochs = trainEpochs

        # Don't want to write over
        self.data = self.loadData()
        if not os.path.isdir(self.directory):
            self.makeDirectories()
            self.model.save(self.data['modelFiles'][-1])
        else:
            self.loadModel(-1)



    def makeDirectories(self):
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        if not os.path.isdir(self.getTrainingDataDirectory()):
            os.makedirs(self.getTrainingDataDirectory())
        if not os.path.isdir(self.getModelDirectory()):
            os.makedirs(self.getModelDirectory())

    def getTrainingDataDirectory(self):
        return os.path.join(self.directory, 'training_data')

    def getModelDirectory(self):
        return os.path.join(self.directory, 'models')

    def saveData(self):
        filename = os.path.join(self.directory, 'training_data.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def loadData(self):
        filename = os.path.join(self.directory, 'training_data.pickle')
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return {
            'directory': self.directory,
            'startTime': datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            'gameParameters': {
                'board_dim': self.board_dim,
                'n': self.n
            },
            'agentParameters': {
                'curiosity': self.curiosity,
                'max_depth': self.max_depth,
                'stochasticExploration': self.stochasticExploration,
                'stochasticDecision': self.stochasticDecision,
                'trainEpochs': self.trainEpochs
            },
            'turnOrder': NInARow.turns,
            'modelFiles': [os.path.join(self.getModelDirectory(), "model_start" + ".h5")],
            'games': []
        }

    def train(self, rounds, iterations, learnFromPastRounds=1,
              agent1=None, agent2=None, learnFromAgent1=True, learnFromAgent2=True):
        for i in tqdm(range(rounds), desc='training'):
            print("Starting round {0}".format(i))
            self.trainRound(iterations, agent1, agent2, learnFromAgent1, learnFromAgent2)
            self.trainModel(learnFromPastRounds*iterations)

    def trainRound(self, iterations, agent1=None, agent2=None, learnFromAgent1=True, learnFromAgent2=True):
        data = self.data

        for i in tqdm(range(iterations), desc='round'):
            gameData = {}
            gameData['startTime'] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            gameData['modelFile'] = self.data['modelFiles'][-1]
            gameData['turns'] = []
            if not agent1:
                agent1 = LearnerAgent(self.model, self.curiosity, self.max_depth, self.stochasticExploration, self.stochasticDecision)
            if not agent2:
                agent2 = LearnerAgent(self.model, self.curiosity, self.max_depth, self.stochasticExploration,
                             self.stochasticDecision)
            if i%2 == 0:
                agents = [agent1, agent2]
                skips = [learnFromAgent1, learnFromAgent2]
            else:
                agents = [agent2, agent1]
                skips = [learnFromAgent2, learnFromAgent1]
            board = NInARow(self.board_dim, self.n)
            currentPlayer = 0
            while board.getResult() == gameInterface.ONGOING:
                player = board.getTurn()
                skip = not skips[currentPlayer]
                move = agents[currentPlayer].getMove(board)
                gameData['turns'].append({'player': player, 'skip': skip, 'action': (int(move.x), int(move.y))})
                board.play(move)
                currentPlayer = (currentPlayer + 1) % len(agents)
            gameData['turns'].append({'player': NInARow.WHITE, 'skip': False, 'action': None})
            gameData['turns'].append({'player': NInARow.BLACK, 'skip': False, 'action': None})
            result = board.getResult()
            if result == NInARow.turns[0]:
                gameData['result'] = (0, 1, 0)
            elif result == NInARow.turns[1]:
                gameData['result'] = (0, 0, 1)
            else:
                gameData['result'] = (1, 0, 0)
            gameData['endTime'] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            data['games'].append(gameData)

        self.saveData()
        return data

    def trainModel(self, lastNGames):
        data = NInARowData(self.data)
        inputs, value_outputs, policy_outputs = data.vectorizeData(np.arange(max(0, len(self.data['games'])-lastNGames), len(self.data['games'])))
        self.model.train(inputs, value_outputs, policy_outputs, self.trainEpochs)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = os.path.join(self.getModelDirectory(), "model_" + now + ".h5")
        self.data['modelFiles'].append(filename)
        self.model.save(filename)
        self.saveData()

    def loadModel(self, index):
        filename = self.data['modelFiles'][index]
        self.model.load(filename)
        return self.model

def getBestAction(tree, iterations):
    for i in range(iterations):
        tree.select(tree.root)
    return tree.getBestAction()

def runGame(board, agent1, agent2):
    agents = [agent1, agent2]
    currentPlayer = 0
    while board.getResult() == board.ONGOING:
        board.play(agents[currentPlayer].getMove(board))
        currentPlayer = (currentPlayer + 1) % len(agents)
    return board.getResult()

def compareModels(gameGenerator, model1, model2, numGames, curiosity, maxNodes):
    model1start = []
    model2start = []
    for i in tqdm(range(numGames)):
        model1start.append(runGame(gameGenerator(),
                                   LearnerAgent(model1, curiosity, maxNodes, True, False),
                                   LearnerAgent(model2, curiosity, maxNodes, True, False)))
        model2start.append(runGame(gameGenerator(),
                                   LearnerAgent(model2, curiosity, maxNodes, True, False),
                                   LearnerAgent(model1, curiosity, maxNodes, True, False)))
    game = gameGenerator()
    # print(model1start)
    # print(model2start)
    model1wins = len([a for a in model1start if a == game.turns[0]]) + len(
        [a for a in model2start if a == game.turns[1]])
    model2wins = len([a for a in model1start if a == game.turns[1]]) + len(
        [a for a in model2start if a == game.turns[0]])
    draws = numGames*2 - model1wins - model2wins
    return {
        'model1start': model1start,
        'model2start': model2start,
        'model1wins': model1wins,
        'model2wins': model2wins,
        'draws': draws
    }



if __name__ == '__main__':
    pass
    # comparisons =
    # board = NInARow(3, 3)
    # board.play((1, 2))
    # board.play((1, 0))
    # model = RandomModel()
    # curiosity = np.sqrt(2)
    # tree = MCTSTree(board, model, curiosity)
    # action = getBestAction(tree, 500)
    # print(action)
    # print(tree.root.values)
    # print(tree.root.getChildren()[0].values)

    # board = NInARow(3, 3)
    # model = NIARKerasFeedForwardModel([256, 256])
    # model.initialize()
    # trainer = NInARowTrainer("../data/ninarow/tictactoe4", 3, 3, np.sqrt(2), 25, model,
    #                          trainEpochs=100,
    #                          stochasticExploration=True, stochasticDecision=True)
    # for i in range(3):
    #     print("Training iteration {0}".format(i+1))
    #     trainer.train(1, 3)
    #     testTicTacToeModel(trainer.model)

    # model.load('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/tictactoe/models/goldStandard256.h5')
    # model = NIARFeedForward(3, 3, [256, 256])
    # model.initialize()
    # trainer = NInARowTrainer("../data/ninarow/tictactoe", 3, 3, np.sqrt(2), 50, model)
    # trainer.train(100, 1)

    # model = NIARFeedForward(3, 3, [256])
    # model.load('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/tictactoe/models/model_20180402125152.h5')
    # board = NInARow(3, 3)
    # value, policy = model.getValueAndPolicy(board, NInARow.WHITE)
    # print(value)
    # print(policy)
    # print(board.getLegalMoves())
    #
    # board.play((0, 2))
    # value, policy = model.getValueAndPolicy(board, NInARow.BLACK)
    # print(value)
    # print(policy)
    # print(board.getLegalMoves())

    # trainer.trainRound(2)
    # data = NInARowData('../data/ninarow/tictactoe/training_data/data_20180328224558.json')
    # inputs, outputs = data.vectorizeData()
    # model.train(inputs, outputs)
    # model.save('../data/ninarow/tictactoe/models/model_20180328224558.h5')
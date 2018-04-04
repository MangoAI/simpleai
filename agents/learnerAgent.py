from agent import Agent
import numpy as np
import sys
sys.path.insert(0, '..')
from games import gameInterface, NInARow
import os
import datetime
from randomAgent import RandomAgent
import json
import math
from models.model import NIARSingleLayer, NInARowData, NIARFeedForward

class State:

    def __init__(self, board):
        self.board = board

class MCTSTree:

    def __init__(self, rootBoard, model, curiosity):
        """
        :param model: function that takes a State and player (turn) and returns a tuple (value, policyDistribution)
        """
        self.root = Node(self, rootBoard)
        self.turn = rootBoard.getTurn()
        self.model = model
        self.curiosity = curiosity
        self.boardToNode = {self.root.boardHash: self.root}
        self.root.getChildren() # initializes children

    def select(self, node):
        if node.visits == 0 or not node.getChildren(): # Has never been visited before or leaf
            node.visits += 1
            return node
        child = node.getChildren()[np.argmax(node.getUCTs(self.curiosity))]
        deepestNode = self.select(child)
        for turn in board.getTurnOrder():
            node.values[turn] = (node.visits * node.getValue(turn) + deepestNode.getValue(turn)) / (node.visits + 1)

        node.visits += 1
        return deepestNode

    def getBestAction(self):
        return max(self.root.getChildren(), key=lambda c: c.getValue(self.turn)).board.getHistory()[-1]

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
        values = np.array([child.getValue(turn) for child in self.getChildren()])
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

    def __init__(self, model, curiosity, maxNodes):
        self.model = model
        self.curiosity = curiosity
        self.maxNodes = maxNodes
        self.tree = None

    def getMove(self, board):
        self.tree = MCTSTree(board, self.model, self.curiosity)
        for i in range(self.maxNodes):
            self.tree.select(self.tree.root)
        return self.tree.getBestAction()

class NInARowTrainer:

    def __init__(self, directory, board_dim, n, curiosity, max_depth, model):
        self.startTime = datetime.datetime.now()
        self.directory = directory
        self.board_dim = board_dim
        self.n = n
        self.curiosity = curiosity
        self.max_depth = max_depth
        self.model = model

        self.makeDirectories()

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

    def train(self, rounds, iterations):
        for i in range(rounds):
            print("Starting round {0}".format(i))
            data = self.trainRound(iterations)
            self.trainModel(data)

    def prepareData(self):
        return {
            'directory': self.directory,
            'startTime': datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            'gameParameters': {
                'board_dim': self.board_dim,
                'n': self.n
            },
            'agentParameters': {
                'curiosity': self.curiosity,
                'max_depth': self.max_depth
            },
            'turnOrder': NInARow.turns,
            'games': []
        }

    def trainRound(self, iterations):
        data = self.prepareData()
        filename = os.path.join(self.getTrainingDataDirectory(), "data_" + data['startTime'] + ".json")

        for i in range(iterations):
            gameData = {}
            gameData['startTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gameData['turns'] = []
            agents = [LearnerAgent(self.model, self.curiosity, self.max_depth),
                      LearnerAgent(self.model, self.curiosity, self.max_depth)]
            board = NInARow(self.board_dim, self.n)
            currentPlayer = 0
            while board.getResult() == gameInterface.ONGOING:
                player = board.getTurn()
                skip = False
                move = agents[currentPlayer].getMove(board)
                gameData['turns'].append({'player': player, 'skip': skip, 'action': (int(move.x), int(move.y))})
                board.play(move)
                currentPlayer = (currentPlayer + 1) % len(agents)
            result = board.getResult()
            if result == NInARow.turns[0]:
                gameData['result'] = (1, -1)
            elif result == NInARow.turns[1]:
                gameData['result'] = (-1, 1)
            else:
                gameData['result'] = (0, 0)
            gameData['endTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data['games'].append(gameData)

        with open(filename, 'w') as f:
            json.dump(data, f)

        return data

    def trainModel(self, trainData):
        data = NInARowData(trainData)
        self.model.train(data)
        filename = os.path.join(self.getModelDirectory(), "model_" + trainData['startTime'] + ".h5")
        self.model.save(filename)

class RandomModel:
    def getValueAndPolicy(self, board, player):
        n = len(board.getLegalMoves())
        return np.random.rand() / 10000, np.ones(n) / n

def getBestAction(tree, iterations):
    for i in range(iterations):
        tree.select(tree.root)
    return tree.getBestAction()

if __name__ == '__main__':
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

    board = NInARow(3, 3)
    model = NIARFeedForward(3, 3, [256])
    model.initialize()
    trainer = NInARowTrainer("../data/ninarow/tictactoe", 3, 3, np.sqrt(2), 500, model)
    trainer.train(100, 1)

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
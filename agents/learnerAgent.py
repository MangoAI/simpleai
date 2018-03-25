from .agent import Agent
import numpy as np
import sys
sys.path.insert(0, '..')
from games import gameInterface, NInARow
import os
import datetime
from .randomAgent import RandomAgent
import json
import math

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
        turn = node.board.getTurn()
        node.values[turn] = (node.visits * node.getValue(turn) + deepestNode.getValue(turn)) / (node.visits + 1)
        node.visits += 1
        return node

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
                value, policy = self.tree.model(self.board, player)
                self.values[player] = value
            else:
                if result == gameInterface.DRAW:
                    self.values = {turn: 0 for turn in self.board.turns}
                else:
                    self.values = {turn: 1 if turn == player else -1 for turn in self.board.turns}
        return self.values[player]

    def getPolicy(self):
        if self.policy is None:
            self.policy = self.tree.model(self.board, self.board.getTurn())
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

def randomModel(board, player):
    n = len(board.getLegalMoves())
    # return 0, np.ones(n)/n
    return np.random.rand()/10000, np.ones(n)/n

class NInARowTrainer:

    def __init__(self, directory, board_dim, n, curiosity, max_depth, model):
        self.startTime = datetime.datetime.now()
        self.directory = os.path.join(directory, self.startTime.strftime("%Y%m%d%H%M%S"))
        self.board_dim = board_dim
        self.n = n
        self.curiosity = curiosity
        self.max_depth = max_depth
        self.model = model

        agentData = {
            'startTime': self.startTime.strftime("%Y-%m-%d %H:%M:%S"),
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
        now = datetime.datetime.now()
        filename = now.strftime(os.path.join(self.directory, "trainingData", "%Y%m%d%H%M%S"))
        data = {
            'startTime': now.strftime("%Y-%m-%d %H:%M:%S"),
            'games': []
        }

        for i in range(iterations):
            gameData = {}
            gameData['startTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gameData['turns'] = []
            agents = [LearnerAgent(randomModel, np.sqrt(2), 50), RandomAgent()]
            board = NInARow(3, 3)
            currentPlayer = 0
            while board.getResult() == gameInterface.ONGOING:
                player = board.getTurn()
                skip = currentPlayer != 0
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
            data['games'].append(gameData)
            print(i, gameData['result'])

        with open(os.path.join(self.directory, 'blah.json'), 'w') as f:
            json.dump(data, f)


class AgentTrainer:

    def __init__(self, directory, agentParameters, model, gameClass, gameParameters):
        # self.directory =
        self.model = model
        self.gameClass = gameClass
        self.gameParameters = gameParameters

        # os.makedirs(self.directory)

    def trainRound(self):
        data = {
            'startTime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'gameParameters': {
                'board_dim': 3,
                'n': 3
            },
            'agentParameters': {
                'curiosity': math.sqrt(2),
                'maxDepth': 50
            },
            'turnOrder': NInARow.turns,
            'games': []
        }


    def train(self):
        data = {
            'game'              : 'n_in_a_row',
            'startTime'         : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'gameParameters'    : {
                                    'board_dim' : 3,
                                    'n'         : 3
                                  },
            'agentParameters'   : {
                                    'curiosity': math.sqrt(2),
                                    'maxDepth': 50
                                  },
            'turnOrder'         : NInARow.turns,
            'games'             : []
        }


        for i in range(100):
            gameData = {}
            gameData['startTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gameData['turns'] = []
            agents = [LearnerAgent(randomModel, np.sqrt(2), 50), RandomAgent()]
            board = NInARow(3, 3)
            currentPlayer = 0
            while board.getResult() == gameInterface.ONGOING:
                player = board.getTurn()
                skip = currentPlayer != 0
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
            data['games'].append(gameData)
            print(i, gameData['result'])

        with open(os.path.join(self.directory, 'blah.json'), 'w') as f:
            json.dump(data, f)

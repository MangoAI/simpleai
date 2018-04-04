from abc import ABC, abstractmethod
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.callbacks import EarlyStopping
import json
import sys
sys.path.insert(0, '../..')
from games import NInARow
import os
import random

class NInARowGame:
    def __init__(self):
        self.turns = []
        self.result = []

class NInARowTurn:
    def __init__(self, player, skip, action):
        self.player = player
        self.skip = skip
        self.action = action

class NInARowParameters:
    def __init__(self, board_dim, n):
        self.board_dim = board_dim
        self.n = n

class LearnerAgentParameters:
    def __init__(self, curiosity, maxDepth):
        self.curiosity = curiosity
        self.maxDepth = maxDepth

class NInARowData:

    def __init__(self, data):
        board_dim = data['gameParameters']['board_dim']
        n = data['gameParameters']['n']
        curiosity = data['agentParameters']['curiosity']
        maxDepth = data['agentParameters']['max_depth']

        self.gameParameters = NInARowParameters(board_dim, n)
        self.agentParameters = LearnerAgentParameters(curiosity, maxDepth)
        self.turnOrder = data['turnOrder']
        self.games = []
        for gameData in data['games']:
            game = NInARowGame()
            game.result = gameData['result']
            for turn in gameData['turns']:
                game.turns.append(NInARowTurn(turn['player'], turn['skip'], turn['action']))
            self.games.append(game)

    @staticmethod
    def loadFromFile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return NInARowData(data)

    def vectorizeData(self):
        inputs = []
        value_outputs = []
        policy_ouputs = []

        for game in self.games:
            board = NInARow(self.gameParameters.board_dim, self.gameParameters.n)
            for turn in game.turns:
                if not turn.skip:
                    inputs.append(board.featurize())
                    value_outputs.append(game.result)
                    policy_ouputs.append(self.vectorizeAction(turn.action))
                board.play(turn.action)
        return np.array(inputs), np.array(value_outputs), np.array(policy_ouputs)

    def vectorizeAction(self, action):
        x, y = action[0], action[1]
        space = np.zeros(self.gameParameters.board_dim**2 + 1) # + 1 for game over, that is no-op
        space[x*self.gameParameters.board_dim + y] = 1
        return space

    def vectorizeNoOp(self):
        space = np.zeros(self.gameParameters.board_dim ** 2 + 1)
        space[-1] = 1
        return space


class KerasModel:

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def getValueAndPolicy(self, board, player):
        pass

    def train(self, data):
        if self.model is None:
            self.initialize()
        inputs, outputs = data.vectorizeData()
        self.model.fit(inputs, outputs, validation_split=.2, batch_size=32, nb_epoch=100)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)

class NInARowKerasModel(KerasModel):

    def __init__(self, board_dim, n):
        self.board_dim = board_dim
        self.n = n
        self.inputDim = 1 + 2 * (self.board_dim**2)
        self.outputDim = 2 + self.board_dim**2

    def getValueAndPolicy(self, board, player):
        input = np.array([board.featurize()])
        prediction = self.model.predict(input)
        values, policy = np.array(prediction[0][0]), np.array(prediction[1][0])
        value = values[1] if player == NInARow.WHITE else values[2]
        legalMoves = [move.x*self.board_dim + move.y for move in board.getLegalMoves()]
        policy = policy[legalMoves]
        return value, policy

class NIARSingleLayer(NInARowKerasModel):

    def __init__(self, board_dim, n, hidden_nodes):
        super().__init__(board_dim, n)
        self.model = None
        self.hidden_nodes = hidden_nodes

    def initialize(self):
        self.model = Sequential()
        self.model.add(Dense(units=self.hidden_nodes, activation='sigmoid', input_dim=self.inputDim))
        self.model.add(Dense(units=self.outputDim, activation='softmax'))
        self.model.compile(loss=categorical_crossentropy,
                      optimizer=Adam())

class NIARFeedForward(NInARowKerasModel):

    def __init__(self, board_dim, n, hidden_nodes):
        super().__init__(board_dim, n)
        self.model = None
        self.hidden_nodes = hidden_nodes

    def initialize(self):
        input = Input(shape=(self.inputDim,), dtype='float32', name='input')
        hidden = Dense(units=self.hidden_nodes[0], activation='sigmoid', input_dim=self.inputDim)(input)
        for h in self.hidden_nodes[1:]:
            hidden = Dense(units=h, activation='sigmoid')(hidden)
        value_output = Dense(2, activation='tanh', name='value_output')(hidden)
        policy_output = Dense(self.board_dim**2, activation='softmax', name='policy_output')(hidden)
        model = Model(inputs=[input], outputs=[value_output, policy_output])
        model.compile(optimizer=Adam(),
                      loss={'value_output': mean_squared_error,
                            'policy_output': categorical_crossentropy})

        self.model = model

    def train(self, data):
        if self.model is None:
            self.initialize()
        inputs, value_outputs, policy_outputs = data.vectorizeData()
        self.model.fit({'input': inputs},
                       {'value_output': value_outputs, 'policy_output': policy_outputs},
                       validation_split=.2, batch_size=32, epochs=100)

class TicTacToeModel(NInARowKerasModel):

    def __init__(self, hidden_nodes):
        super().__init__(3, 3)
        self.model = None
        self.hidden_nodes = hidden_nodes

    def initialize(self):
        input = Input(shape=(self.inputDim,), dtype='float32', name='input')
        hidden = Dense(units=self.hidden_nodes[0], activation='sigmoid', input_dim=self.inputDim)(input)
        for h in self.hidden_nodes[1:]:
            hidden = Dense(units=h, activation='sigmoid')(hidden)
        value_output = Dense(3, activation='softmax', name='value_output')(hidden)
        policy_output = Dense(self.board_dim**2+1, activation='softmax', name='policy_output')(hidden)
        model = Model(inputs=[input], outputs=[value_output, policy_output])
        model.compile(optimizer=Adam(),
                      loss={'value_output': categorical_crossentropy,
                            'policy_output': categorical_crossentropy})

        self.model = model

    def train(self):
        if self.model is None:
            self.initialize()
        inputs, value_outputs, policy_outputs = TicTacToeModel.vectorizeScoresWithAllActions()
        self.model.fit({'input': inputs},
                       {'value_output': value_outputs, 'policy_output': policy_outputs},
                       batch_size=256, epochs=1000)

    @staticmethod
    def vectorizeAction(action):
        x, y = action[0], action[1]
        space = np.zeros(10)
        space[x*3 + y] = 1
        return space

    @staticmethod
    def vectorizeNoop():
        space = np.zeros(10)
        space[-1] = 1
        return space

    @staticmethod
    def vectorizeScoresWithAllActions():
        valuesAndActions = NInARow.loadValuesAndActions()
        boards, values, policies = [], [], []
        for board in valuesAndActions:
            value = valuesAndActions[board]['value']
            actions = valuesAndActions[board]['actions']
            featurizedBoard = board.featurize()
            if value == 0:
                vectorizedValues = (1, 0, 0)
            elif value == 1:
                vectorizedValues = (0, 1, 0)
            else:  # value == -1
                vectorizedValues = (0, 0, 1)

            if actions:
                for action in actions:
                    boards.append(featurizedBoard)
                    values.append(vectorizedValues)
                    policies.append(TicTacToeModel.vectorizeAction(action))
            else:
                boards.append(featurizedBoard)
                values.append(vectorizedValues)
                policies.append(TicTacToeModel.vectorizeNoop())
        return np.array(boards), np.array(values), np.array(policies)


if __name__ == '__main__':
    # NInARow.createPickles()
    # inputs, values, policies = TicTacToeModel.vectorizeScoresWithRandomAction()
    # model = NIARFeedForward(10, 5, [64, 64, 64])
    # model.initialize()
    model = TicTacToeModel([256, 256])
    model.train()
    model.save('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/tictactoe/models/goldStandard256_256.h5')
    # model.load('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/tictactoe/models/goldStandard256.h5')
    pass

    # 256: val loss: ~0.003, policy loss: ~ .85 after 2000 epochs
    # 512 (1000 epochs): us / step - loss: 0.9128 - value_output_loss: 0.0058 - policy_output_loss: 0.9070

    # model = NInARowLogRegModel(3, 3)
    # data = NInARowData('../data/blah.json')
    # inputs, outputs = data.vectorizeData()
    # model.initialize()
    # model.train(inputs, outputs)
    # model.save('tttlogreg.h5')
    # model.load('tttlogreg.h5')
    # game = NInARow(3, 3)
    # value, policy = model.getValueAndPolicy(game, NInARow.WHITE)
    # print(game.getLegalMoves()[np.argmax(policy)])
    # game.play((1,1))
    # game.play((1,0))
    # print(game)
    # value, policy = model.getValueAndPolicy(game, NInARow.WHITE)
    # print(game.getLegalMoves()[np.argmax(policy)])
    # game.play((0, 2))
    # game.play((0, 1))
    # value, policy = model.getValueAndPolicy(game, NInARow.WHITE)
    # print(game)
    # print(game.getLegalMoves()[np.argmax(policy)])
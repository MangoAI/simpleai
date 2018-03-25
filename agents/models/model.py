# from abc import ABC, abstractmethod
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import load_model
from keras.callbacks import EarlyStopping
import json
import sys
sys.path.insert(0, '../..')
from games import NInARow
import os

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

    def __init__(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        board_dim = data['gameParameters']['board_dim']
        n = data['gameParameters']['n']
        curiosity = data['agentParameters']['curiosity']
        maxDepth = data['agentParameters']['maxDepth']

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

    def vectorizeData(self):
        inputs = []
        outputs = []

        for game in self.games:
            board = NInARow(self.gameParameters.board_dim, self.gameParameters.n)
            for turn in game.turns:
                if not turn.skip:
                    inputs.append(board.featurize())
                    outputs.append(np.append(game.result, self.vectorizeAction(turn.action)))
                board.play(turn.action)
        return np.array(inputs), np.array(outputs)

    def vectorizeAction(self, action):
        x, y = action[0], action[1]
        space = np.zeros(self.gameParameters.board_dim**2)
        space[x*self.gameParameters.board_dim + y] = 1
        return space

class Model:

    pass


class KerasModel(Model):

    def load(self, saveFile):
        pass

class NInARowKerasModel(KerasModel):

    def __init__(self, board_dim, n):
        self.board_dim = board_dim
        self.n = n
        self.inputDim = 1 + 2 * (self.board_dim**2)
        self.outputDim = 2 + self.board_dim**2
#
class NInARowLogRegModel(NInARowKerasModel):

    def __init__(self, board_dim, n):
        super().__init__(board_dim, n)
        self.model = None

    def initialize(self):
        self.model = Sequential()
        self.model.add(Dense(units=self.board_dim**2, activation='sigmoid', input_dim=self.inputDim))
        # self.model.add(Dense(units=self.board_dim**2, activation='sigmoid'))
        self.model.add(Dense(units=self.outputDim, activation='softmax'))
        self.model.compile(loss=categorical_crossentropy,
                      optimizer=Adam())

    def train(self, inputs, outputs):
        if self.model is None:
            self.initialize()
        self.model.fit(inputs, outputs, validation_split=.2, batch_size=32, nb_epoch=100)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)

    def getValueAndPolicy(self, board, player):
        input = np.array([board.featurize()])
        prediction = self.model.predict(input)[0]
        values = prediction[:2]
        value = values[0] if player == NInARow.WHITE else values[1]
        policy = prediction[2:]
        legalMoves = [move.x*self.board_dim + move.y for move in board.getLegalMoves()]
        policy = policy[legalMoves]
        print(value)
        print(policy)
        return value, policy


if __name__ == '__main__':
    model = NInARowLogRegModel(3, 3)
    # data = NInARowData('../data/blah.json')
    # inputs, outputs = data.vectorizeData()
    # model.initialize()
    # model.train(inputs, outputs)
    # model.save('tttlogreg.h5')
    model.load('tttlogreg.h5')
    game = NInARow(3, 3)
    value, policy = model.getValueAndPolicy(game, NInARow.WHITE)
    print(game.getLegalMoves()[np.argmax(policy)])
    game.play((1,1))
    game.play((1,0))
    print(game)
    value, policy = model.getValueAndPolicy(game, NInARow.WHITE)
    print(game.getLegalMoves()[np.argmax(policy)])
    game.play((0, 2))
    game.play((0, 1))
    value, policy = model.getValueAndPolicy(game, NInARow.WHITE)
    print(game)
    print(game.getLegalMoves()[np.argmax(policy)])
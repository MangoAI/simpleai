from abc import abstractmethod
import numpy as np
from keras.models import Model, Sequential, load_model, clone_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, mean_squared_error
import collections
import json
import sys
sys.path.insert(0, '../..')
from games import NInARow
from tqdm import tqdm


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

    def vectorizeData(self, indices):
        inputs = []
        value_outputs = []
        policy_ouputs = []
        for game in tqdm([self.games[i] for i in indices], desc='Vectorizer'):
            board = NInARow(self.gameParameters.board_dim, self.gameParameters.n)
            for turn in game.turns:
                if not turn.skip:
                    inputs.append(board.featurize())
                    value_outputs.append(game.result)
                    # value_outputs.append(self.vectorizeResult(game.result))
                    policy_ouputs.append(self.vectorizeAction(turn.action))
                if turn.action:
                    board.play(turn.action)
        return np.array(inputs), np.array(value_outputs), np.array(policy_ouputs)

    def vectorizeResult(self, result):
        assert result is NInARow.WHITE or result is NInARow.BLACK or result is NInARow.DRAW
        if result == NInARow.DRAW:
            vectorizedValues = (1, 0, 0)
        elif result == NInARow.WHITE:
            vectorizedValues = (0, 1, 0)
        else: # result == NInARow.BLACK
            vectorizedValues = (0, 0, 1)
        return np.array(vectorizedValues)

    def vectorizeAction(self, action):
        """
        Last dimension is no-op, i.e. Game is over, do nothing
        :param action:
        :return:
        """
        space = np.zeros(self.gameParameters.board_dim ** 2 + 1)
        if action:
            x, y = action[0], action[1]
            space[x * self.gameParameters.board_dim + y] = 1
        else:
            space[-1] = 1
        return space

class RandomModel:
    def getValueAndPolicy(self, board, player):
        n = len(board.getLegalMoves())
        return np.random.rand() / 10000, np.ones(n) / n

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


class NInARowModel:
    def __init__(self, board_dim, n):
        self.board_dim = board_dim
        self.n = n
        self.inputDim = 1 + 2 * (self.board_dim ** 2)
        self.outputValueDim = 3
        self.outputPolicyDim = 1 + self.board_dim ** 2
        self.model = None

    def vectorizeResult(self, result):
        assert result is NInARow.WHITE or result is NInARow.BLACK or result is NInARow.DRAW
        if result == NInARow.DRAW:
            vectorizedValues = (1, 0, 0)
        elif result == NInARow.WHITE:
            vectorizedValues = (0, 1, 0)
        else: # result == NInARow.BLACK
            vectorizedValues = (0, 0, 1)
        return np.array(vectorizedValues)

    def vectorizeAction(self, action):
        """
        Last dimension is no-op, i.e. Game is over, do nothing
        :param action:
        :return:
        """
        space = np.zeros(self.board_dim ** 2)
        if action:
            x, y = action[0], action[1]
            space[x * self.board_dim + y] = 1
        else:
            space[-1] = 1
        return space

    def vectorizeNoop(self):
        return self.vectorizeAction(None)

    def getValuesAndPolicies(self, boards, players):
        if not isinstance(players, collections.Iterable) or isinstance(players, str):
            players = [players]*len(boards)
        assert len(boards) == len(players)
        input = np.array([board.featurize() for board in boards])
        prediction = self.model.predict(input)
        values, policies = np.array(prediction[0]), np.array(prediction[1])
        whiteValues = values.T[1] - values.T[2]
        values = [whiteValues[i] if players[i] == NInARow.WHITE else -1*whiteValues[i] for i in range(len(boards))]
        legalPolicies = []
        for i in range(len(boards)):
            legalMoves = [move.x*self.board_dim + move.y for move in boards[i].getLegalMoves()]
            legalPolicy = np.array(policies[i][legalMoves])
            legalPolicy /= np.sum(legalPolicy)
            legalPolicies.append(legalPolicy)
        return np.array(values), legalPolicies

    def getValueAndPolicy(self, board, player):
        values, policies = self.getValuesAndPolicies([board], [player])
        return values[0], policies[0]

    def getBestChildrenValues(self, board, player):
        moves = board.getLegalMoves()
        children = [board.copy().play(move) for move in moves]
        values, policies = self.getValuesAndPolicies(children, player)
        return {moves[i]: values[i] for i in range(len(moves))}

class NIARKerasFeedForwardModel(NInARowModel):

    def __init__(self, board_dim, n, hidden_nodes):
        super().__init__(board_dim, n)
        self.hidden_nodes = hidden_nodes

    def initialize(self):
        input = Input(shape=(self.inputDim,), dtype='float32', name='input')
        hidden = Dense(units=self.hidden_nodes[0], activation='sigmoid', input_dim=self.inputDim)(input)
        for h in self.hidden_nodes[1:]:
            hidden = Dense(units=h, activation='sigmoid')(hidden)
        value_output = Dense(self.outputValueDim, activation='softmax', name='value_output')(hidden)
        policy_output = Dense(self.outputPolicyDim, activation='softmax', name='policy_output')(hidden)
        model = Model(inputs=[input], outputs=[value_output, policy_output])
        model.compile(optimizer=Adam(),
                      loss={'value_output': categorical_crossentropy,
                            'policy_output': categorical_crossentropy})
        self.model = model

    def train(self, board_feature_vector, value_vectors, policy_vectors, epochs):
        """
        :param boards: list of NINARow boards
        :param results: list of results for each board, result must be in {NInARow.WHITE, NInARow.BLACK, NInARow.DRAW}
        :param moves: list of move tuples in (x, y)
        :return:
        """
        if self.model is None:
            self.initialize()

        self.model.fit({'input': board_feature_vector},
                       {'value_output': value_vectors, 'policy_output': policy_vectors},
                       verbose=False,
                       batch_size=256, epochs=epochs)

    def clone(self):
        model = NIARKerasFeedForwardModel(self.board_dim, self.n, self.hidden_nodes)
        model.model = clone_model(self.model)
        model.model.set_weights(self.model.get_weights())
        model.model.compile(optimizer=Adam(),
                      loss={'value_output': categorical_crossentropy,
                            'policy_output': categorical_crossentropy})
        return model

    def load(self, filename):
        self.model = load_model(filename)

    def save(self, filename):
        self.model.save(filename)

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

    def getValuesAndPolicies(self, boards, players):
        assert len(boards) == len(players)
        input = np.array([board.featurize() for board in boards])
        prediction = self.model.predict(input)
        values, policies = np.array(prediction[0]), np.array(prediction[1])
        whiteValues = values.T[1] - values.T[2]
        values = [whiteValues[i] if players[i] == NInARow.WHITE else -1*whiteValues[i] for i in range(len(boards))]
        legalPolicies = []
        for i in range(len(boards)):
            legalMoves = [move.x*self.board_dim + move.y for move in boards[i].getLegalMoves()]
            legalPolicies.append(np.array(policies[i][legalMoves]))
        return np.array(values), legalPolicies

    def getValueAndPolicy(self, board, player):
        values, policies = self.getValuesAndPolicies([board], [player])
        return values[0], policies[0]

class PerfectTicTacToe(NInARowModel):

    def __init__(self):
        super().__init__(3, 3)
        self.valuesAndActions = NInARow.loadValuesAndActions()

    def getValueAndPolicy(self, board, player):
        value = self.valuesAndActions[board]['value']
        value = -value if player == NInARow.BLACK else value
        actions = self.valuesAndActions[board]['actions']
        policy = np.mean([self.vectorizeAction(action) for action in actions], axis=0)
        legalMoves = [move.x * self.board_dim + move.y for move in board.getLegalMoves()]
        return value, policy[legalMoves]




def getBestAction(board, policy):
    return board.getLegalMoves()[np.argmax(policy)]

def testTicTacToeModel(model):
    valuesAndActions = NInARow.loadValuesAndActions()
    valueDistance = 0
    correctActions = 0
    actionsTaken = 0
    confidenceDistance = 0
    boards = list(valuesAndActions.keys())
    values, policies = model.getValuesAndPolicies(boards, [NInARow.WHITE]*len(boards))
    for i in range(len(boards)):
        board = boards[i]
        actualValue = valuesAndActions[board]['value']
        predictedValue = values[i]
        valueDistance += np.abs(actualValue - predictedValue)
        actualActions = valuesAndActions[board]['actions']
        if actualActions:
            predictedAction = getBestAction(board, policies[i])
            actionIsCorrect = (predictedAction.x, predictedAction.y) in actualActions
            correctActions += actionIsCorrect
            if actionIsCorrect:
                confidenceDistance += np.abs((1/len(actualActions)) - np.max(policies[i]))
            actionsTaken += 1
    avgValueDistance = valueDistance/len(boards)
    correctActionPercentage = correctActions/actionsTaken
    avgConfidenceDistance = confidenceDistance/correctActions
    # print("Total number of boards evaluated: {0}".format(len(boards)))
    # print("Average value distance: {0}".format(np.round(avgValueDistance, 4)))
    # print("Correct actions: {0} out of {1} ({2}%)".format(
    #     correctActions, actionsTaken,
    #     np.round(correctActionPercentage, 4)))
    # print("Average confidence: {0}".format(np.round(avgConfidenceDistance, 4)))
    return {
        'totalBoards': len(boards),
        'avgValueDistance': avgValueDistance,
        'correctActionPercentage': correctActionPercentage,
        'avgConfidenceDistance': avgConfidenceDistance,
        'correctActions': correctActions,
        'actionsTaken': actionsTaken
    }




if __name__ == '__main__':
    # NInARow.createPickles()
    # inputs, values, policies = TicTacToeModel.vectorizeScoresWithRandomAction()
    # model = NIARFeedForward(10, 5, [64, 64, 64])
    # model.initialize()
    model = TicTacToeModel([256, 256])
    model.load('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/tictactoe/models/goldStandard256.h5')
    # model.load('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/tictactoe/models/goldStandard256_256.h5')
    testTicTacToeModel(model)
    # b1 = NInARow(3, 3
    # b2 = b1.copy().play((1, 1))
    # b3 = b2.copy().play((0, 1))
    # b4 = b3.copy().play((0, 0))
    # b5 = b4.copy().play((2, 2))
    # values, policies = model.getValuesAndPolicies([b1, b2, b3,b4,b5],
    #         [NInARow.WHITE, NInARow.BLACK, NInARow.WHITE, NInARow.BLACK, NInARow.WHITE])

    # model.train()
    # model.save('/Users/a.nam/Desktop/mangoai/simpleai/data/ninarow/tictactoe/models/goldStandard256_256.h5')
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
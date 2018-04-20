import pickle
import numpy as np
from abc import abstractmethod
from utils import parseDateTime
from data.agentParameters import AgentParameters

class TurnData:

    def __init__(self, player, action, skipTraining):
        self.player = player
        self.action = action
        self.skipTraining = skipTraining

class GameRoundData:

    def __init__(self, startTime, endTime, modelFilename, turns, result):
        self.startTime = startTime
        self.endTime = endTime
        self.modelFilename = modelFilename
        self.result = result
        self.turns = turns

    def getWinner(self):
        """
        Returns the index of winning player according to self.result
        Assumes that self.result is a vector
        Assumes that the highest score wins
        :return:
        """
        return np.argmax(self.result)

class GameData:

    def __init__(self, filename):
        self.filename = filename

        with open(self.filename, 'rb') as f:
            data = pickle.load(f)
        self.startTime = parseDateTime(data['startTime'])
        self.turnOrder = data['turnOrder']
        self.gameParameters = self.loadGameParameters(data['gameParameters'])
        self.agentParameters = self.loadAgentParameters(data['agentParameters'])
        self.modelsFilenames = data['modelFiles']
        self.games = self.loadGames(data['games'])

    @abstractmethod
    def loadGameParameters(self, gameParameters):
        pass

    def loadAgentParameters(self, agentParameters):
        return AgentParameters(
            curiosity=agentParameters['curiosity'],
            max_depth=agentParameters['max_depth'],
            stochasticExploration=agentParameters['stochasticExploration'],
            stochasticDecision=agentParameters['stochasticDecision'],
            trainEpochs=agentParameters['trainEpochs']
        )

    def loadGames(self, rawGameData):
        games = []
        for game in rawGameData:
            games.append(GameRoundData(
                startTime=parseDateTime(game['startTime']),
                endTime=parseDateTime(game['endTime']),
                modelFilename=game['modelFile'],
                turns=[TurnData(turn['player'], turn['action'], turn['skip']) for turn in game['turns']],
                result=game['result']
            ))
        return games
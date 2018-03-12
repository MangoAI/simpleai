from abc import ABC, abstractmethod
import numpy as np
from threading import Thread, Event
from boardInterface import LOSE_SCORE, ONGOING
import time

class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def getMove(self, board):
        pass

class RandomAgent(Agent):

    def __init__(self):
        pass

    def getMove(self, board):
        return np.random.choice(list(board.getLegalMoves()))

class MinMaxAgent(Agent):
    def __init__(self, turnOrder,
                 heuristicFunction, maxDepth, maxThinkingTime):
        """
        :param turnOrder: list of player colors in turn order, starting with self
            e.g. [NInARow.WHITE, NInARow.BLACK] if player is WHITE
            e.g. [NInARow.BLACK, NInARow.WHITE] if player is BLACK
        :param heuristicFunction: takes board, returns scalar
        :param maxDepth: int in [1, inf)
        :param maxThinkingTime: float in (0, inf)
        """
        self.turnOrder = turnOrder
        self.heuristicFunction = heuristicFunction
        self.maxDepth = maxDepth
        self.maxThinkingTime = maxThinkingTime
        self.stopEvent = None
        self.bestAction = None
        self.memory = None

    def memoryHeuristic(self, board, turn):
        key = (board, turn)
        if key not in self.memory:
            self.memory[key] = self.heuristicFunction(board, turn)
        return self.memory[key]

    def getMove(self, board):
        self.memory = {}
        self.stopEvent = Event()
        self.action_thread = Thread(target=self.minMax, args=[board])

        # Here we start the thread and we wait 5 seconds before the code continues to execute.
        self.action_thread.start()
        self.action_thread.join(timeout=self.maxThinkingTime)
        self.stopEvent.set()

        while self.bestAction is None:
            time.sleep(.05)
        return self.bestAction

    def minMax(self, board):
        self.bestAction = None
        self.bestAction, bestBoard = self.getMax(board, self.turnOrder, self.maxDepth)
        self.stopEvent.set()
        # print()

    def getMax(self, board, turnOrder, depth):
        if depth == 0 or self.stopEvent.is_set() or board.getResult() is not ONGOING:
            return (None, board)

        legalMoves = board.getLegalMoves()
        nextBoards = []
        for move in legalMoves:
            nextBoards.append(board.copy())
            nextBoards[-1].play(move)
        nextTurnOrder = turnOrder[1:] + [turnOrder[0]]
        nextBoards = [self.getMax(nextBoard, nextTurnOrder, depth-1)[1] for nextBoard in nextBoards]
        bestAction, bestBoard, bestScore = None, None, LOSE_SCORE
        for nextAction, nextBoard in zip(legalMoves, nextBoards):
            score = self.memoryHeuristic(nextBoard, turnOrder[0]) + (np.random.rand()/10000)
            if score >= bestScore:
                bestAction, bestBoard, bestScore = nextAction, nextBoard, score
            if self.stopEvent.is_set():
                break
        return bestAction, bestBoard

class MCTSAgent(Agent):

    def __init__(self, maxThinkingTime):
        pass

    def getMove(self, board):
        while True:

    def select(self, board):

    def search(self, board):
        result = board.getResult()
        if result is not ONGOING:

        if game.gameEnded(s): return -game.gameReward(s)

        if s not in visited:
            visited.add(s)
            P[s], v = nnet.predict(s)
            return -v

        max_u, best_a = -float("inf"), -1
        for a in range(game.getValidActions(s)):
            u = Q[s][a] + c_puct * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a

        sp = game.nextState(s, a)
        v = search(sp, game, nnet)

        Q[s][a] = (N[s][a] * Q[s][a] + v) / (N[s][a] + 1)
        N[s][a] += 1
        return -v

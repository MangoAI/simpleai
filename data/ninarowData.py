from data.gameData import GameData

class NInARowParameters:

    def __init__(self, board_dim, n):
        self.board_dim = board_dim
        self.n = n

class NInARowData(GameData):
    def __init__(self, filename):
        super().__init__(filename)

    def loadGameParameters(self, gameParameters):
        return NInARowParameters(gameParameters['board_dim'], gameParameters['n'])
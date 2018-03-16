from agents import RandomAgent, MinMaxAgent
from games import Chess, ONGOING, NInARow
from games.ninarow.board import Move

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

if __name__ == '__main__':
    wins = 0
    for i in range(1, 101):
        game = Game(NInARow(3, 3), MinMaxAgent(NInARow.turns, lambda a, b: 0, 5, 100), RandomAgent())
        game.board.play(Move(1, 0, 2))
        game.board.play(Move(-1, 1, 0))
        result = game.play()
        if result == NInARow.turns[0]:
            wins += 1
        print(i, wins/i)
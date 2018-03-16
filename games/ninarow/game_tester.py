import numpy as np
from .ninarow import WHITE, BLACK, ONGOING, getOtherPlayer
from .board import Board, Move

def testGetResult():
    board = Board(16, 5)
    board.play(Move(WHITE,0, 0)) # WHITE
    board.play(Move(BLACK,0, 1)) # BLACK
    board.play(Move(WHITE,1, 0)) # WHITE
    assert board.getResult() == ONGOING
    board.play(Move(BLACK,0, 2)) # BLACK
    board.play(Move(WHITE,2, 0)) # WHITE
    board.play(Move(BLACK,0, 3)) # BLACK
    board.play(Move(WHITE,3, 0)) # WHITE
    assert board.getResult() == ONGOING
    board.play(Move(BLACK,0, 4)) # BLACK
    board.play(Move(WHITE,1, 1)) # WHITE
    board.play(Move(BLACK,0, 5)) # BLACK
    assert board.getResult() == BLACK

    board = Board(16, 5)
    board.play(Move(WHITE,0, 0))      # WHITE
    board.play(Move(BLACK,15, 15))    # BLACK
    board.play(Move(WHITE,1, 0))      # WHITE
    board.play(Move(BLACK,15, 14))    # BLACK
    board.play(Move(WHITE,2, 0))      # WHITE
    board.play(Move(BLACK,14, 15))    # BLACK
    board.play(Move(WHITE,3, 0))      # WHITE
    board.play(Move(BLACK,14, 14))    # BLACK
    board.play(Move(WHITE,0, 1))      # WHITE
    board.play(Move(BLACK,15, 13))    # BLACK
    board.play(Move(WHITE,0, 2))      # WHITE
    board.play(Move(BLACK,14, 13))    # BLACK
    board.play(Move(WHITE,0, 3))      # WHITE
    board.play(Move(BLACK,13, 13))    # BLACK
    board.play(Move(WHITE,1, 1))      # WHITE
    board.play(Move(BLACK,13, 14))    # BLACK
    board.play(Move(WHITE,1, 2))      # WHITE
    board.play(Move(BLACK,13, 15))    # BLACK
    assert board.getResult() == ONGOING
    board.play(Move(WHITE,4, 0))      # WHITE
    assert board.getResult() == WHITE

    board = Board(16, 5)
    board.play(Move(WHITE,0, 0))      # WHITE
    board.play(Move(BLACK,15, 15))    # BLACK
    board.play(Move(WHITE,1, 1))      # WHITE
    board.play(Move(BLACK,15, 14))    # BLACK
    board.play(Move(WHITE,2, 2))      # WHITE
    board.play(Move(BLACK,14, 15))    # BLACK
    board.play(Move(WHITE,4, 4))      # WHITE
    board.play(Move(BLACK,14, 14))    # BLACK
    board.play(Move(WHITE,3, 3))      # WHITE
    assert board.getResult() == WHITE

    board = Board(16, 5)
    board.play(Move(WHITE,4, 4))  # WHITE
    board.play(Move(BLACK,15, 15))  # BLACK
    board.play(Move(WHITE,3, 3))  # WHITE
    board.play(Move(BLACK,15, 14))  # BLACK
    board.play(Move(WHITE,2, 2))  # WHITE
    board.play(Move(BLACK,14, 15))  # BLACK
    board.play(Move(WHITE,1, 1))  # WHITE
    board.play(Move(BLACK,14, 14))  # BLACK
    board.play(Move(WHITE,0, 0))  # WHITE
    assert board.getResult() == WHITE

    board = Board(16, 5)
    board.play(Move(WHITE,0, 0))  # WHITE
    board.play(Move(BLACK,15, 15))  # BLACK
    board.play(Move(WHITE,1, 1))  # WHITE
    board.play(Move(BLACK,15, 14))  # BLACK
    board.play(Move(WHITE,2, 2))  # WHITE
    board.play(Move(BLACK,14, 15))  # BLACK
    board.play(Move(WHITE,4, 4))  # WHITE
    board.play(Move(BLACK,3, 3))  # BLACK
    board.play(Move(WHITE,3, 5))  # WHITE
    assert board.getResult() == ONGOING

def testPlay():
    board_dim = 8
    n = 4
    board = Board(board_dim, n)
    moves = np.random.choice(board_dim*board_dim, 20, replace=False)
    moves = [Move(WHITE if i%2 is 0 else BLACK, moves[i]%board_dim, moves[i]//board_dim) for i in range(len(moves))]

    turn = WHITE
    for move in moves:
        board.play(move)
        assert board.getPiece(move.x, move.y) == turn
        turn = getOtherPlayer(turn)
    for move in moves:
        assert board.getPiece(move.x, move.y) == turn
        turn = getOtherPlayer(turn)

def getLegalMoves():
    board_dim = 8
    n = 4
    board = Board(board_dim, n)
    allPossibleMoves = set([(i%board_dim, i//board_dim) for i in range(board_dim**2)])

    moves = np.random.choice(board_dim * board_dim, 20, replace=False)
    moves = [Move(WHITE if i%2 is 0 else BLACK, moves[i]%board_dim, moves[i]//board_dim) for i in range(len(moves))]
    for move in moves:
        board.play(move)
        allPossibleMoves.remove((move.x, move.y))
        if board.getResult() is not ONGOING:
            break
        assert {(move.x, move.y) for move in board.getLegalMoves()} == allPossibleMoves


if __name__ == '__main__':
    testGetResult()
    testPlay()
    getLegalMoves()
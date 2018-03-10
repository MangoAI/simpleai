class NInARow:
    WHITE = 1
    BLACK = -1
    BLANK = 0

    DRAW = 0
    ONGOING = 2

    @staticmethod
    def getOtherPlayer(color):
        return color * -1

if __name__ == "__main__":
    pass
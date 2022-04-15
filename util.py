import numpy as np

def toBinary(num):
    if num < 0:
        raise SystemExit(1)
    elif num < 2:
        return str(num)
    else:
        return toBinary(num // 2) + str(num % 2)

def printBoard(board):
    rows = []
    for j in range(len(board)):
        row = ''
        for i in range(len(board[j])):
            row += 'x' if board[j][i] == 1 else '0'
        rows.append(row)
    rows.reverse()
    print("--")
    for r in rows:
        print(r)
    print("--")

def decodeBoard(num):
    binNum = toBinary(num)
    if len(binNum) < 16:
        binNum = ('0' * (16 - len(binNum))) + binNum
    board = np.zeros((4,4))
    for i in range(0,4):
        for j in range(0,4):
            index = i*4 + j
            if binNum[index] == '0':
                board[i,j] = 0
            else:
                board[i,j] = 1
    return board

def printQTable(qtable):
    print("--- START QTABLE ---\n")
    for tileType in qtable:
        for state in tileType:
            printBoard(decodeBoard(state))
            for action in tileType[state]:
                print("Action: " + str(action) + ", Reward: " + str(tileType[state][action]))
    print("\n--- END QTABLE ---")

# Encode the board state into a binary number
def encode_boardstate(gameboard):
    binaryString = ''
    for r in range(gameboard.N_row):
        for c in range(gameboard.N_col):
            if gameboard.board[r,c] == 1:
                binaryString += '1'
            else:
                binaryString += '0'
    return int(binaryString, base=2)



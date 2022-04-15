import numpy as np
import matplotlib.pyplot as plt

def toBinary(num):
    if num < 0:
        raise SystemExit(1)
    elif num < 2:
        return str(num)
    else:
        return toBinary(num // 2) + str(num % 2)

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

#  [
#    { 0: {(1, 0): 0.0}
#    , 24576: {(2, 0): 0.0}
#    , 25344: {(3, 0): 0.0}
#    , 25392: {(2, 0): 0.0}
#    , 25395: {(2, 0): -100.0}
#    }
#  ]

def printQTable(qtable):
    for tileType in qtable:
        for state in tileType:
            printBoard(decodeBoard(state))
            for action in tileType[state]:
                print("Action: " + str(action))
                print("Reward: " + str(tileType[state][action]))

rewards = np.load('rewards.npy')
qtable = np.load('qtable.npy', allow_pickle=True)

plt.figure()
plt.plot(rewards)
plt.show()

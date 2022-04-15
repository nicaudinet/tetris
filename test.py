import numpy as np
import random
import unittest
from gameboardClass import TGameBoard
from agentClass import TQAgent

from debug import *

class TestQTable(unittest.TestCase):
    
    def setUp(self):

        # param_set == PARAM_TASK1a:

        # Create the agent
        alpha = 0.2
        epsilon = 0
        episode_count = 1000
        self.agent = TQAgent(alpha,epsilon,episode_count)

        # Create the gameboard
        self.N_row = 4
        self.N_col = 4
        tile_size = 2
        max_tile_count = 50
        stochastic_prob = 0
        self.gameboard = TGameBoard(
                self.N_row,
                self.N_col,
                tile_size,
                max_tile_count,
                self.agent,
                stochastic_prob)

        # Set the possible tiles for the gameboard
        self.gameboard.tiles = [
            [[[0,2]]],
            #  [[[0,2]], [[0,1],[0,1]]],
            [[[0,2],[0,2]]] # Square tile
                #  [[[0,2],[1,2]], [[0,2],[0,1]], [[0,1],[0,2]], [[1,2],[0,2]]]
        ]

        # Create predefined tile sequence, used if stochastic_prob=0
        self.gameboard.tile_sequence = []
        for x in range(max_tile_count):
            randomNumber = random.randint(0,len(self.gameboard.tiles)-1)
            self.gameboard.tile_sequence.append(randomNumber)

        # Initialize the agent
        self.gameboard.agent.fn_init(self.gameboard)
        self.gameboard.fn_restart()


    def test_emptyBoard(self):
        board = self.agent.gameboard.board
        emptyBoard = np.array([
            [-1,-1,-1,-1],
            [-1,-1,-1,-1],
            [-1,-1,-1,-1],
            [-1,-1,-1,-1]
            ])
        np.testing.assert_array_equal(board, emptyBoard)
        self.assertEqual(self.agent.boardstate, 0)

    #  def test_bla(self):
    #      printBoard(self.agent.gameboard.board)
    #      self.agent.fn_turn()
    #      printBoard(self.agent.gameboard.board)
    #      self.agent.fn_turn()
    #      printBoard(self.agent.gameboard.board)
    #      self.agent.fn_turn()
    #      printBoard(self.agent.gameboard.board)
    #      self.agent.fn_turn()
    #      printBoard(self.agent.gameboard.board)
    #      self.agent.fn_turn()
    #      printBoard(self.agent.gameboard.board)
    #      self.agent.fn_turn()
    #      printBoard(self.agent.gameboard.board)
    #      self.agent.fn_turn()

    def test_finalReward(self):
        with self.assertRaises(SystemExit):
            old_episode = self.agent.episode
            while True:
                if self.agent.episode != old_episode:
                    #  printQTable(self.agent.qtable)
                    old_episode = self.agent.episode
                self.agent.fn_turn()
        rewardRange = range(self.agent.episode-1,self.agent.episode)
        reward = np.sum(self.agent.reward_tots[rewardRange])
        self.assertEqual(reward, 250)


if __name__ == '__main__':
    unittest.main()

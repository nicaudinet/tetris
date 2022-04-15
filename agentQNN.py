import numpy as np
import random
import math
import h5py
import copy
import torch

class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count


    # Instructions:
    # In this function you could set up and initialize the states, actions,
    # the Q-networks (one for calculating actions and one target network),
    # experience replay buffer and storage for the rewards
    # You can use any framework for constructing the networks, for example
    # pytorch or tensorflow
    # This function should not return a value, store Q network etc as attributes of self
    def fn_init(self,gameboard):
        self.gameboard=gameboard

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the
        # experience replay buffer

        self.boardstate = encode_boardstate(self.gameboard)
        self.tile_type = self.gameboard.cur_tile_type
        self.action = (self.gameboard.tile_x, self.gameboard.tile_orientation) 
        self.reward_tots = np.zeros(self.episode_count)

        # FIXME: Add some NN shit


    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file


    def fn_read_state(self):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board
        # and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row
        # 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that
        # should be placed on the game board (integer between 0 and
        # len(self.gameboard.tiles))

    def fn_select_action(self):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for
        # the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an
        # attribute of self and exectute the action by moving the tile to the
        # desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number
        # where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to
        # execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <=
        # tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree
        # rotations of the tile (0 < tile_orientation < # of non-degenerate
        # rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self,batch):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last
        # action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the
        # Q-network to calculate the values Q(s_old,a), i.e. the estimate of the
        # future reward for all actions a
        # Then repeat for the target network to calculate the value \hat
        # Q(s_new,a) of the new state (use \hat Q=0 if the new state is
        # terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                reward = np.sum(self.reward_tots[range(self.episode-100,self.episode)])
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(reward),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the
            # variable 'old_state' which is later stored in the ecperience
            # replay buffer

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the
            # total reward for the current episode, so you can save it to disk
            # later

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch'
                # containing 'self.batch_size' quadruplets 
                self.fn_reinforce(batch)
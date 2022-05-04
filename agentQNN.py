import numpy as np
import random
import math
import h5py
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from util import encode_boardstate


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)

    def push(self, state):
        """Save a transition"""
        self.memory.append(state)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def is_full(self):
        return len(self.memory) == self.capacity

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, gameboard):

        super(DQN, self).__init__()

        self.num_tiles = len(gameboard.tiles)

        num_inputs = gameboard.N_row * gameboard.N_col + len(gameboard.tiles)
        num_outputs = gameboard.N_col * 4

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, num_outputs)
        )


    def format_input(self, board, tile_type):
        board = np.ndarray.flatten(board)
        tile_type_one_hot = np.zeros(self.num_tiles)
        tile_type_one_hot[tile_type] = 1
        # Using floats here because by default the weights of the network use floats
        return torch.tensor(np.append(board, tile_type_one_hot), dtype=torch.float)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, board, tile_type):
        inputs = self.format_input(board, tile_type)
        return self.layers(inputs)




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

        self.action = (self.gameboard.tile_x, self.gameboard.tile_orientation) 
        self.reward_tots = np.zeros(self.episode_count)

        self.exp_buffer = ReplayMemory(self.replay_buffer_size)

        # Used to calculate the next action
        # Updated at every change of state
        self.nn_calc = DQN(gameboard)

        # Target network kept constant for long periods of time
        # Used to evaluate performance
        # Synchronized to nn_calc every sync_target_episode_count episodes
        self.nn_target = copy.deepcopy(self.nn_calc)


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

        actions = self.nn_calc.forward(self.gameboard.board, self.gameboard.cur_tile_type)
        actions = actions.detach().numpy()
        
        while True:
            max_action_index = np.argmax(actions)
            tile_position = max_action_index // 4
            tile_orientation = max_action_index % 4
            move = self.gameboard.fn_move(tile_position, tile_orientation)
            if move == 0:
                self.action = (tile_position, tile_orientation)
                break
            else:
                actions[max_action_index] = np.NINF

        # FIXME Implement the epsilon thingy here as well


    def fn_reinforce(self, batch):
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

        loss = nn.MSELoss()
        optimizer = optim.Adam(self.nn_calc.parameters())

        for quadruplet in batch:

            old_state, action, reward, new_state = quadruplet

            optimizer.zero_grad()
            actions = self.nn_calc(self.gameboard)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            actions = self.nn_calc.forward(self.gameboard).detach().numpy()

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                reward = np.sum(self.reward_tots[range(self.episode-100,self.episode)])
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(reward),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # Here you can save the rewards and the Q-network to data files
                    reward_filename = 'qnn_rewards/' + str(self.episode) + '.npy'
                    np.save(reward_filename, self.reward_tots)
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                buffer_long_enough = len(self.exp_buffer) >= self.replay_buffer_size 
                sync_target_episode = (self.episode % self.sync_target_episode_count) == 0
                if buffer_long_enough and sync_target_episode:
                    # Here you should write line(s) to copy the current network to the target network
                    self.nn_target = copy.deepcopy(self.nn_calc)
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()

            # Here you should write line(s) to copy the old state into the
            # variable 'old_state' which is later stored in the experience
            # replay buffer
            old_state = (self.gameboard.board, self.gameboard.cur_tile_type)

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            # Here you should write line(s) to add the current reward to the
            # total reward for the current episode, so you can save it to disk
            # later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # Here you should write line(s) to store the state in the experience replay buffer
            new_state = (self.gameboard.board, self.gameboard.cur_tile_type)
            state = (old_state, self.action, reward, new_state)
            self.exp_buffer.push(state)

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # Here you should write line(s) to create a variable 'batch'
                # containing 'self.batch_size' quadruplets 
                batch = self.exp_buffer.sample(self.batch_size)
                self.fn_reinforce(batch)

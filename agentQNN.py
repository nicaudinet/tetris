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


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        return self.layers(state)




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


    # Set up and initialize the states, actions, the Q-networks, experience
    # replay buffer and storage for the rewards
    def fn_init(self,gameboard):
        self.gameboard=gameboard

        self.action = (self.gameboard.tile_x, self.gameboard.tile_orientation) 

        self.reward_tots = np.zeros(self.episode_count)

        self.exp_buffer = ReplayMemory(self.replay_buffer_size)

        # Find all possible actions
        possible_actions = {}
        for tile_idx, tile in enumerate(self.gameboard.tiles):
            tile_actions = {}
            for orientation in range(len(tile)):
                num_positions = self.gameboard.N_col + 1 - len(tile[orientation])
                tile_actions[orientation] = num_positions
            possible_actions[tile_idx] = tile_actions
        self.possible_actions = possible_actions

        # Neural Network initailizations
        self.nn_calc = DQN(gameboard)
        self.nn_target = copy.deepcopy(self.nn_calc)
        self.optimizer = optim.Adam(self.nn_calc.parameters())
        self.criterion = nn.MSELoss()


    # Load the Q-network (to Q-network of self) from the strategy_file
    def fn_load_strategy(self,strategy_file):
        pass


    # Calculate the current state of the gane board
    def fn_read_state(self):
        self.board = self.gameboard.board.flatten()
        self.tile_type = self.gameboard.cur_tile_type
        num_tiles = len(self.gameboard.tiles)
        tile = np.zeros(num_tiles)
        tile[self.tile_type] = 1
        self.state = torch.tensor(np.hstack([self.board, tile]), dtype=torch.float)


    # Choose and execute an action, based on the output of the Q-network for
    # the current state, or random if epsilon greedy
    def fn_select_action(self):

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number
        # where epsilon_N changes from unity to epsilon

        curr_e_scale = 1 - self.episode / self.epsilon_scale
        curr_e = np.max([self.epsilon, curr_e_scale])

        tile_actions = self.possible_actions[self.tile_type]
        if np.random.rand() < curr_e:
            # Choose random move
            tile_orientation = np.random.randint(0, len(tile_actions))
            tile_position = np.random.randint(0, tile_actions[tile_orientation])
            self.action = (tile_position, tile_orientation)
            self.gameboard.fn_move(tile_position, tile_orientation)
        else:
            # Choose the move with the highest expected reward
            with torch.no_grad():
                output = self.nn_calc(self.state).detach().numpy()
            while True:
                max_action_index = np.argmax(output)
                tile_position = max_action_index // self.gameboard.N_col
                tile_orientation = max_action_index % self.gameboard.N_col
                move = self.gameboard.fn_move(tile_position, tile_orientation)
                if move == 0:
                    self.action = (tile_position, tile_orientation)
                    break
                else:
                    output[max_action_index] = np.NINF


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
        # This function should not return a value, the Q table is stored as an
        # attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to
        # update the Q-network

        loss = nn.MSELoss()

        for quadruplet in batch:

            old_state, action, reward, new_state = quadruplet

            # Get expected rewards for current network
            board = self.gameboard.board
            tile_type = self.gameboard.cur_tile_type
            expected_rewards = self.nn_calc.forward(board, tile_type)
            expected_rewards = expected_rewards.detach().numpy()

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
                es = 'episode ' + str(self.episode) + '/' + str(self.episode_count)
                rs = '(reward: ' + str(reward) + ')'
                print(es + ' ' + rs)
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
                    # Here you should write line(s) to copy the current network
                    # to the target network
                    self.nn_target = copy.deepcopy(self.nn_calc)
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and
            # orientation)
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

            # Here you should write line(s) to store the state in the experience
            # replay buffer
            new_state = (self.gameboard.board, self.gameboard.cur_tile_type)
            state = (old_state, self.action, reward, new_state)
            self.exp_buffer.push(state)

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # Here you should write line(s) to create a variable 'batch'
                # containing 'self.batch_size' quadruplets 
                batch = self.exp_buffer.sample(self.batch_size)
                self.fn_reinforce(batch)

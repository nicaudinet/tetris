import numpy as np
import random
import math
import h5py
import copy

import debug

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    # Instructions:
    # In this function you could set up and initialize the states, actions
    # and Q-table and storage for the rewards
    # This function should not return a value, store Q table etc as
    # attributes of self
    def fn_init(self,gameboard):
        self.gameboard=gameboard
        self.boardstate = self.encode_boardstate()
        self.tile_type = self.gameboard.cur_tile_type
        self.action = (self.gameboard.tile_x, self.gameboard.tile_orientation) 
        self.reward_tots = np.zeros(self.episode_count)
        # type QTable = Map TileType (Map State (Map Action ExpectedReward))
        self.qtable = [ {} for i in range(len(gameboard.tiles)) ]


    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input
        # parameter strategy_file (used to test how the agent plays)


    # Encode the board state into a binary number
    def encode_boardstate(self):
        binaryString = ''
        for r in range(self.gameboard.N_row):
            for c in range(self.gameboard.N_col):
                if self.gameboard.board[r,c] == 1:
                    binaryString += '1'
                else:
                    binaryString += '0'
        return int(binaryString, base=2)
    

    # Return the list of possible actions and their rewards for a given tile type
    def possible_actions(self, tile_type, boardstate):

        # Find all possible actions
        tiles = self.gameboard.tiles[tile_type]
        actions = []
        for orientation in range(len(tiles)):
            num_positions = self.gameboard.N_col + 1 - len(tiles[orientation])
            for position in range(num_positions):
                actions.append((position, orientation))

        # Match possible actions with their expected rewards
        qtable = self.qtable[tile_type]
        actionRewards = {}
        for action in actions:
            # If action wasn't seen before assume 0 reward
            actionRewards[action] = 0.0
            if boardstate in qtable:
                if action in qtable[boardstate]:
                    actionRewards[action] = qtable[boardstate][action]

        return actionRewards


    # Instructions:
    # In this function you could calculate the current state of the gane board
    # You can for example represent the state as an integer entry in the Q-table
    # This function should not return a value, store the state as an attribute of self
    def fn_read_state(self):
        self.boardstate = self.encode_boardstate()
        self.tile_type = self.gameboard.cur_tile_type


    # Instructions:
    # Choose and execute an action, based on the Q-table or random if epsilon greedy
    # This function should not return a value, store the action as an
    # attribute of self and exectute the action by moving the tile to the
    # desired position and orientation

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
    def fn_select_action(self):

        # Find possible next actions
        qtable = self.qtable[self.tile_type]
        possible_actions = self.possible_actions(self.tile_type, self.boardstate)

        if self.boardstate not in qtable:

            # If the state was not seen before choose random action
            self.action = random.choice(list(possible_actions))

        else:

            # Categorize actions
            bestActions = []
            otherActions = []
            bestAction = max(possible_actions, key=possible_actions.get)
            for action in possible_actions:
                if possible_actions[action] == possible_actions[bestAction]:
                    bestActions.append(action)
                else:
                    otherActions.append(action)
            
            # Choose the next action
            if random.random() < self.epsilon:
                self.action = random.choice(otherActions)
            else:
                self.action = random.choice(bestActions)
        
        # Execute the action
        (tile_x, tile_orientation) = self.action
        self.gameboard.fn_move(tile_x, tile_orientation)

    
    # Instructions:
    # Update the Q table using state and action stored as attributes in self
    # and using function arguments for the old state and the reward
    # This function should not return a value, the Q table is stored as an
    # attribute of self
    def fn_reinforce(self, old_state, reward):

        # Q-Table update formula:
        # Q(s,a) = alpha*[reward + max_a'Q(s',a')] + (1-alpha)*Q(s,a)

        # s = old_state
        # a = old_action
        # a' = new_action
        # s' = new_boardstate

        old_boardstate = old_state[0]
        old_tile_type = old_state[1]
        old_action = self.action
        new_boardstate = self.boardstate

        # Find A = max_a'Q(s',a')
        # Careful! max_a' is the argmax for the best action for _all_ possible
        # tiles, not just the current tile
        A = 0
        for tile_type in range(len(self.gameboard.tiles)):
            qtable = self.qtable[tile_type]
            if new_boardstate in qtable:
                actions = self.possible_actions(tile_type, new_boardstate)
                best_action = max(actions, key=actions.get)
                A = max(A, actions[best_action])

        # Find B = Q(s,a)
        B = 0
        qtable = self.qtable[old_tile_type]
        if old_boardstate in qtable:
            if old_action in qtable[old_boardstate]:
                B = qtable[old_boardstate][old_action]

        # Update the qtable
        C = self.alpha * (reward + A) + (1 - self.alpha) * B
        qtable = self.qtable[old_tile_type]
        if old_boardstate in qtable:
            qtable[old_boardstate][old_action] = C
        else:
            qtable[old_boardstate] = {old_action: C}


    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                reward = np.sum(self.reward_tots[range(self.episode-100,self.episode)])
                print('episode '+str(self.episode)+'/'+str(self.episode_count)
                        +' (reward: ',str(reward),')')
            if self.episode % 1000 == 0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # Here you can save the rewards and the Q-table to data
                    # files for plotting of the rewards and the Q-table can be
                    # used to test how the agent plays
                    np.save('rewards.npy', self.reward_tots)
                    np.save('qtable.npy', self.qtable)
            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()

            # Here you should write line(s) to copy the old state into the
            # variable 'old_state' which is later passed to fn_reinforce()
            old_state = (copy.deepcopy(self.boardstate), copy.deepcopy(self.tile_type))

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            # Here you should write line(s) to add the current reward to the
            # total reward for the current episode, so you can save it to disk
            # later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # Update the Q-table using the old state and the reward (the new
            # state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state, reward)


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

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions,
        # the Q-networks (one for calculating actions and one target network),
        # experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example
        # pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the
        # experience replay buffer

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


class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()

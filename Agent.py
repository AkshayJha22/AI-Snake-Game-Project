import torch
import random
import numpy as np
from collections import deque
from SnakeGame import SnakeGame, Direction, Point
from Model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.no_of_games = 0 # Number of games
        self.epsilon = 0 #ransomness control parameter
        self.gamma = 0.9 # discount rate, has to be < 1
        self.memory = deque(maxlen = MAX_MEMORY) #if memory exceeded then elements are popped from the left

        self.model = Linear_QNet(11, 256, 3) # Input, Hidden and Output layer sizes of NN
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, game):
        head = game.snake[0] # get head from game

        # points next to the head in all directions
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # boolenas to check what current game direction is
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or # if direction moving in has an obstacle in the same direction
            (dir_l and game.is_collision(point_l)) or # check both current direction and if there are any obstacles in that direction
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or # if direction moving in has an obstacle on its right
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or # if direction moving in has an obstacle on its left
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x, # food is to the left
            game.food.x > game.head.x, # food is to the right
            game.food.y < game.head.y, # food is above
            game.food.y > game.head.y # food is below
        ]

        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff b/w exploration and exploitation
        self.epsilon = 80 - self.no_of_games # more games, smaller epsilon, less randomness
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)

            move = torch.argmax(prediction).item() # takes the max argument of predictions and turns it into an int
            final_move[move] = 1 # the max argument of predictions determines the direction change

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGame()

    while True:
        state_old = agent.get_state(game) # get old state
        final_move = agent.get_action(state_old) # get move from old state
        reward, done, score = game.play_step(final_move) #perform move and get new state

        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.no_of_games += 1
            
            # train long memory
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game: ', agent.no_of_games,'Score: ', score,"Record: ", record )

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.no_of_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)
            
if __name__ == '__main__':
    train()
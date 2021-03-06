### Remove goal state, dense, levels, make it finite horizon
### increase character set to add rewards
# Does self.level make contours?
### change the step function, make it more compact
### add self.horizon

import numpy as np
import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import register
import matplotlib.pyplot as plt
import imageio

from envShape import allRooms




class GridRooms(gym.Env):
    """
    Navigational tasks where reaching goal yields a reward
    """
    def __init__(self, roomName='Four-rooms',
                 randomStart=True, randomGoal=True,
                 dense=False):

        if roomName not in allRooms:
            raise TypeError("Invalid room name \'%s\'" % roomName)

        self.randomStart = randomStart
        # self.randomGoal = randomGoal
        self.dense = dense

        self.room = allRooms[roomName]
        self.height = len(self.room)
        self.width = len(self.room[0])

        self.horizon = 20
        self.pause_time = 0.1

        # Set state and action spaces
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.height, self.width, 1), dtype='uint8')

        #  Validate room
        chrs = set()
        for i in range(self.height):
            chrs = chrs.union(set(self.room[i]))

        self.allowedChars = [' ', 'S', '#', 'a', 'b']

        if len(chrs.difference(set(self.allowedChars))) != 0:
            raise TypeError("Invalid values in the grid")

        self.rewardDict = {' ': 0.0, 'S': 0.0, '#': 0.0, 'a': 2.0, 'b': 5.0 }
        self.StartChars = ['S', ' ', 'a', 'b']
        self.rewardChars = ['a', 'b']
        l = len(self.rewardChars)
        self.rewardColours = {'a': (0,2*int(255/l),0), 'b': (0,int(255/l),0)}



        #  Get start states
        self.startStates = []
        self.rewardStates = []
        # self.goalStates = []
        for i in range(self.height):
            for j in range(self.width):
                
                if randomStart is True:
                    if self.room[i][j] in self.StartChars:
                        self.startStates.append((i, j))
                elif self.room[i][j] == 'S':
                    self.startStates.append((i, j))

                if self.room[i][j] in self.rewardChars:
                    self.rewardStates.append((i, j))

                # if randomGoal is True:
                #     if self.room[i][j] in ['S', ' ', 'G']:
                #         self.goalStates.append((i, j))
                # elif self.room[i][j] == 'G':
                #     self.goalStates.append((i, j))

        # if len(self.goalStates) == 0 or len(self.startStates) == 0:
        #     raise ValueError("Either goal or start states not specified")
        if len(self.rewardStates) == 0 or len(self.startStates) == 0:
            raise ValueError("Either reward or start states not specified")

        # goalInd = np.random.randint(len(self.goalStates))
        # self.goal = self.goalStates[goalInd]

        # if dense:
        #     self.__createLevels(self.goal)

        self.done = False
        self.steps = 0
        self.img = None
        self.images = []
        self.__actionSpace()

        if len(self.startStates) == 0:
            self.startStates.append((1, 1))

        self.reset()

    # def __createLevels(self, goal):
    #     stack = [goal]
    #     level = {goal: 0}

    #     while len(stack) > 0:
    #         cur = stack.pop()
    #         x = cur[0]
    #         y = cur[1]
    #         vals = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #         for dx, dy in vals:
    #             if self.room[x + dx][y + dy] != '#':
    #                 pos = (x + dx, y + dy)
    #                 if pos not in level:
    #                     level[pos] = level[cur] - 1
    #                     stack.insert(0, pos)
    #     self.level = level

    def __resetStart(self):

        # if self.randomGoal:
        #     goalInd = np.random.randint(len(self.goalStates))
        #     self.goal = self.goalStates[goalInd]
        # while True:
        #     startInd = np.random.randint(len(self.startStates))
        #     self.start = self.startStates[startInd]
        #     if (self.start != self.goal):
        #         break
        startInd = np.random.randint(len(self.startStates))
        self.start = self.startStates[startInd]


    def __actionSpace(self):
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3

    def __getStateRep(self):
        grid = np.zeros((self.height, self.width, 1))
        grid[self.pos[0], self.pos[1], 0] = 1
        return grid

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        self.steps += 1
        x = self.pos[0]
        y = self.pos[1]
        oldPos = tuple(self.pos)
        # targetPos = tuple(self.pos)

        if action == self.LEFT:
            targetPos = (x - 1, y)
        elif action == self.RIGHT:
            targetPos = (x + 1, y)
        elif action == self.UP:
            targetPos = (x, y - 1)
        elif action == self.DOWN:
            targetPos = (x, y + 1)

        # validate position
        if self.room[targetPos[0]][targetPos[1]] != '#':
            self.pos = targetPos

        # if action == self.LEFT:
        #     if self.room[x - 1][y] != '#':
        #         self.pos = (x - 1, y)
        # elif action == self.RIGHT:
        #     if self.room[x + 1][y] != '#':
        #         self.pos = (x + 1, y)
        # elif action == self.UP:
        #     if self.room[x][y - 1] != '#':
        #         self.pos = (x, y - 1)
        # elif action == self.DOWN:
        #     if self.room[x][y + 1] != '#':
        #         self.pos = (x, y + 1)

        rew = 0.0
        # if self.dense:
        #     rew = (self.level[self.pos] - self.level[oldPos]) * 0.2

        rew = rew + self.rewardDict[self.room[self.pos[0]][self.pos[1]]]

        # if tuple(self.pos) == tuple(self.goal):
        #     rew = 10.0
        #     self.done = True

        if self.steps >= self.horizon:
            #  rew = 10
            self.done = True

        return self.__getStateRep(), rew, self.done, {}

    def reset(self):
        self.__resetStart()
        self.pos = self.start
        self.steps = 0
        self.img = None
        self.done = False
        return self.__getStateRep()

    def render(self, disp=1):
        grid = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if self.room[i][j] != '#':
                    grid[i, j] = 0
                else:
                    grid[i, j] = 1
        grid = 255 - grid * 220
        grid = np.transpose(np.asarray([grid, grid, grid]), [1, 2, 0])
        grid = grid.astype(np.uint8)
        grid[self.start[0], self.start[1]] = (0, 0, 255)
        
        for (i,j) in self.rewardStates:
            grid[i, j] = self.rewardColours[ self.room[i][j] ]
        # grid[self.goal[0], self.goal[1]] = (0, 255, 0)
        grid[self.pos[0], self.pos[1]] = (255, 0, 0)
        if self.img is None:
            plt.axis('off')
            self.img = plt.imshow(grid)
        else:
            self.img.set_data(grid)

        self.images.append(grid)
        if disp != 0:
            plt.pause(self.pause_time)
            plt.draw()
        return grid

    def close(self, savePath=None):
        if savePath:
            imageio.mimsave(savePath, self.images)
        self.images = []
        plt.close()


register(
    id='FourRooms-v0',
    entry_point='gridenv:GridRooms',
    kwargs={'roomName': 'Four-rooms', 'dense': True,
            'randomStart': True, 'randomGoal': False}
)
register(
    id='OneRoom-v0',
    entry_point='gridenv:GridRooms',
    kwargs={'roomName': 'One-room', 'dense': False,
            'randomStart': False, 'randomGoal': False}
)
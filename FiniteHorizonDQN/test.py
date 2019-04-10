# Why does the speed slow down after a while?
# increase fraction of exploration
# Loading a model without an environment, this model cannot be trained until it has a valid environment.
# with random goalState, it doesn't learn properly

import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN
from dqn import DQN

import gridenv
# import gridenv2

NUM_EPISODES = 1
TOTAL_TIMESTEPS = 500000
GAMMA = 0.99
# ROOM_NAME = "FourRooms"
ROOM_NAME = "OneRoom"


# env = gym.make("FourRooms-v0")
env = gym.make(ROOM_NAME+"-v0")
env = DummyVecEnv([lambda: env])

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps = TOTAL_TIMESTEPS )

# model.save("deepq_FourRooms")
model.save("deepq_"+ROOM_NAME)

del model # remove to demonstrate saving and loading

# model = DQN.load("deepq_FourRooms")
model = DQN.load("deepq_"+ROOM_NAME)

print( "Model trained and loaded")

for i in range( NUM_EPISODES ):

  obs = env.reset()
  done = False
  ret = 0
  
  while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    ret = reward + GAMMA * ret

print( str(NUM_EPISODES) + " episodes ended")
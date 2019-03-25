import gym
import gridenv

env = gym.make("FourRooms-v0")
GAMMA = 0.99

done = False
ret = 0
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    #  Will display env
    env.render()
    ret = reward + GAMMA * ret
print("episode ended")
    


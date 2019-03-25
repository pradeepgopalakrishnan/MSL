import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import gridenv

env = gym.make("FourRooms-v0")
GAMMA = 0.99

# multiprocess environment
n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make("FourRooms-v0") for i in range(n_cpu)])
env = DummyVecEnv([lambda: env])

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_FourRooms")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_FourRooms")

done = False
ret = 0

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    ret = rewards + GAMMA * ret
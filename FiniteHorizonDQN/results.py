import matplotlib.pyplot as plt
import pickle
import numpy as np

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

f_name = "total_ep_rew_list.pkl"
f_name_vanilla = "total_ep_rew_list_vanilla.pkl"

with open(f_name, "rb") as f:
    total_ep_rew_list = np.array( pickle.load(f) )

with open(f_name_vanilla, "rb") as f_vanilla:
    total_ep_rew_list_vanilla = np.array( pickle.load(f_vanilla) )

assert len( total_ep_rew_list ) == len( total_ep_rew_list_vanilla ), "Unequal number of training episodes"

l = len( total_ep_rew_list )
num_data_points = 100.0

ep_indices = np.arange( l )
n = 3

sample_list = moving_average( total_ep_rew_list, n )
sample_list_vanilla = moving_average( total_ep_rew_list_vanilla, n )
ep_indices = ep_indices[ n-1: ]

assert len( ep_indices ) == len( sample_list )

sample_list = sample_list[ : : int( l / num_data_points ) ]
sample_list_vanilla = sample_list_vanilla[ : : int( l / num_data_points ) ]
ep_indices = ep_indices[ : : int( l / num_data_points ) ]

cumulative_total_ep_rew = np.sum( total_ep_rew_list )
cumulative_total_ep_rew_vanilla = np.sum( total_ep_rew_list_vanilla )
absolute_gain = cumulative_total_ep_rew - cumulative_total_ep_rew_vanilla
fractional_gain = absolute_gain / cumulative_total_ep_rew_vanilla

print( cumulative_total_ep_rew )
print( cumulative_total_ep_rew_vanilla )
print( absolute_gain )
print( fractional_gain )

# print( len( sample_list ) )
plt.plot( ep_indices, sample_list, "g" )
plt.plot( ep_indices, sample_list_vanilla, "r" )
plt.legend(("Our", "Vanilla"))
plt.title("Comparison our _ vs. Vanilla")
plt.ylabel("Total returns per episode")
plt.xlabel("Number of episodes")
plt.show()


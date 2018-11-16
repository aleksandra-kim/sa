import numpy as np
import time

def generate_trajectory_2(num_parameters, levels=4, lower=0, upper=1):
    """
    Generate a trajectory for ``num_parameters``.
    Returns an array where the rows are steps in the trajectory, 
    and the columns are coordinates for each parameter. 
    The resulting array therefore has dimensions (num_parameters + 1, num_parameters).
    """
    pa = np.tile(np.linspace(lower, upper, levels), num_parameters).reshape((num_parameters, levels))
    starting_indices = np.random.randint(0, high=levels - 1,size=num_parameters)
    start = pa[np.arange(num_parameters), starting_indices]
    end = pa[np.arange(num_parameters), starting_indices - int(levels / 2)]
    order = np.random.permutation(np.arange(num_parameters)).reshape((1, -1))
    indices = np.arange(num_parameters + 1).reshape((-1, 1))
    start_dense = np.tile(start, num_parameters + 1).reshape((-1, num_parameters))
    end_dense = np.tile(end, num_parameters + 1).reshape((-1, num_parameters))
    mask = indices > order
    start_dense[mask] = end_dense[mask]
    return start_dense

n_params = 50000
t1 = time.time()
samples = generate_trajectory_2(n_params, levels=4, lower=0, upper=1)
t2 = time.time()
t = t2-t1

print(samples.shape)
print('Trajectories generated in ' + str(t/60) + ' min')







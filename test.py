import numpy as np
import time
from sys import argv, getsizeof 
from scipy.spatial.distance import cdist

def generate_trajectory_2(num_parameters, levels=4, lower=0, upper=1):
    """
    Generate a trajectory for ``num_parameters``.
    Returns an array where the rows are steps in the trajectory, 
    and the columns are coordinates for each parameter. 
    The resulting array therefore has dimensions (num_parameters + 1, num_parameters).
    """
    #pa = np.tile(np.arange(levels,dtype=np.int8), num_parameters).reshape((num_parameters, levels))
    pa = np.tile(np.linspace(lower, upper, levels, dtype=np.float16), num_parameters).reshape((num_parameters, levels))
    starting_indices = np.random.randint(0, high=levels - 1,size=num_parameters)
    start = pa[np.arange(num_parameters), starting_indices]
    end = pa[np.arange(num_parameters), starting_indices - int(levels / 2)]
    order = np.random.permutation(np.arange(num_parameters)).reshape((1, -1))
    indices = np.arange(num_parameters + 1).reshape((-1, 1))
    start_dense = np.tile(start, num_parameters + 1).reshape((-1, num_parameters))
    end_dense = np.tile(end, num_parameters + 1).reshape((-1, num_parameters))

    del pa, starting_indices, start, end

    mask = indices > order
    change = end_dense[mask]
    del end_dense
    start_dense[mask] = change
    #start_dense[mask] = end_dense[mask]
    del change, mask
    return start_dense

def main(argv):

    n_params = int(argv[0])

    #Samples1
    t1 = time.time()
    samples1 = generate_trajectory_2(n_params, levels=4, lower=0, upper=1)
    t2 = time.time()
    t = t2-t1

    print('Samples1 \n Trajectories generated in ' + str(t/60) + ' min')
    print('Memory: ' + str(samples1.shape[0]) + '-by-' + str(samples1.shape[1]) + ' array --> ' + str(getsizeof(samples1)) + ' bytes \n')
        #'mask - ' + str(mask.shape[0]) + '-by-' + str(mask.shape[1]) + ' array --> ' + str(getsizeof(mask)) + ' bytes' )

    #Samples2
    t1 = time.time()
    samples2 = generate_trajectory_2(n_params, levels=4, lower=0, upper=1)
    t2 = time.time()
    t = t2-t1

    print('Samples2 \n Trajectories generated in ' + str(t/60) + ' min')
    print('Memory: ' + str(samples2.shape[0]) + '-by-' + str(samples2.shape[1]) + ' array --> ' + str(getsizeof(samples2)) + ' bytes \n')

    #Distance between trajectories
    distance = np.array(np.sum(cdist(samples1, samples2)), dtype=np.float16)


if __name__ == "__main__":
   main(argv[1:])



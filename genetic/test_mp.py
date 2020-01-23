# prime_mutiprocessing.py

import time
import math
from multiprocessing import Pool
from multiprocessing import freeze_support


'''Define function to run mutiple processors and pool the results together'''
def run_multiprocessing(func, args, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(func, args)


'''Define task function'''
def is_prime(args):
    n, a = args
    if (n < 2) or (n % 2 == 0 and n > 2):
        return False
    elif n == 2:
        return True
    elif n == 3:
        return True
    else:
        for i in range(3, math.ceil(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True


def main():
    start = time.perf_counter()

    '''
    set up parameters required by the task
    '''
    num_max = 1000000
    n_processors = 8
    args = []
    for i in range(num_max):
        args.append((i, 3))
    x_ls = list(range(num_max))

    '''
    pass the task function, followed by the parameters to processors
    '''
    out = run_multiprocessing(is_prime, args, n_processors)
    print("Input length: {}".format(len(x_ls)))
    print("Output length: {}".format(len(out)))
    print("Mutiprocessing time: {} mins\n".format((time.perf_counter()-start)/60))
    print("Mutiprocessing time: {} secs\n".format((time.perf_counter()-start)))


if __name__ == "__main__":
    freeze_support()   # required to use multiprocessing
    main()
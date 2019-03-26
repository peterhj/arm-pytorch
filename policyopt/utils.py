import numpy as np
import random
import time
import timeit

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)

def perf_counter():
    try:
        return time.perf_counter()
    except AttributeError:
        return timeit.default_timer()

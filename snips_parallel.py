# Parallel programming with Pool

# Importing the multiprocessing 
from multiprocessing import Pool

# function to which we'll perform multi-processing
def cube(i):
    i = i+1
    z = i**3
    return z

# using pool class to map the function with iterable arguments-
print(Pool().map(cube, [1, 2, 3]))


#----------------------------------------------------------------

''' use threading module for paralel running of some function '''

import time
from threading import Thread

def no_arg(func, instances): # func is function withOUT arguments
    for i in range(instances): # number of threads equals instances
        t = Thread(target=func)
        t.start()

def with_arg(func, instances,args): # func is function with arguments
    for i in range(instances): # number of threads equals instances
        t = Thread(target=func, args = args) # arguments in tuple
	t.start()


#-----------------------------------------------------------------

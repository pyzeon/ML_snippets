
help(len)
len? # help on an object

L = [1, 2, 3]
L?

square?? # reading the source code

L.<TAB> # see a list of all available attributes of an object
L.c<TAB>
from itertools import co<TAB>
import <TAB> # see which imports are available


str.*find*? # looking for a string method that contains the word find somewhere

%run myscript.py # running external code

%timeit L = [n ** 2 for n in range(1000)]


# -----------------------------------------------------------------------------------
# chained comparison with all kind of operators
a = 10
print(1 < a < 50)
print(10 == a < 20)



# -----------------------------------------------------------------------------------
# calling different functions with same arguments based on condition

def product(a, b):
    return a * b
def subtract(a, b):
    return a - b

b = True
print((product if b else subtract)(1, 1))


# -----------------------------------------------------------------------------------

def ls(dir="."):
    """ List files in current directory or directory passed as parameter """
    import os # Inside the method to avoid it being added to global namespace
    return [os.path.abspath(dir), os.listdir(dir)]



# --------------------------------------------------------------------

""" You can have an 'else' clause with try/except. 
    It gets excecuted if no exception is raised.
    This allows you to put less happy-path code in the 'try' block so you can be 
    more sure of where a caught exception came from."""

try:
    1 + 1
except TypeError:
    print("Oh no! An exception was raised.")
else:
    print("Oh good, no exceptions were raised.")

#--------------------------------------------------------------------------------------------------------
# else gets called when for loop does not reach break statement
a = [1, 2, 3, 4, 5]
for el in a:
    if el == 0:
        break
else:
     print('did not break out of for loop')


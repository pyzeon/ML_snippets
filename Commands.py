
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


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

help('modules') # check which modules are installed
# -----------------------------------------------------------------------------------

import sys
sys.path
# When you ask Python to import a module, ...
# ... it starts with the first directory in sys.path ...
# ... and checks for an appropriate file. 
# If no match is found in the first directory it checks subsequent entries, in order, until a match is found 

# To create a normal module ...
# ... you simply createa Python source file in a directory contained in sys.path. 
# The process for creating packages is not much different. 


# -----------------------------------------------------------------------------------

# usually only the last row from a cell is shown
# the command below swithces it to show all the outputcommands from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# -----------------------------------------------------------------------------------

import pandas as pd
import sys
colors = pd.Series(['periwinkle','mint green','burnt orange','periwinkle',
                    'burnt orange','rose','rose','mint green','rose','navy'])
colors.apply(sys.getsizeof) # show the memory occupied by each individual value
# Keep in mind these are Python objects that have some overhead in the first place. 
# ... (sys.getsizeof('') will return 49 bytes.)

colors.memory_usage() # sums up the memory usage
                      # relies on the .nbytes attribute of the underlying NumPy array


sys.getsizeof(colors)


# -----------------------------------------------------------------------------------
# Pointers

# In Python the variables are just labels (kind of pointers), not containers (as in C)
a = [1,2,3]
b = a
c = b
a[1] = 5 # here the 2nd elmnt of "b" and "c" will also change. It wouldn't be the case if variable were containers
print(a,b,c)

# This Python feature causes that one variable could be assigned to different types
x = "Hello"
print(x)
x = 5
print(x)


# (is vs. ==) is checks if two variables refer to the same object, 
# but == checks if the objects pointed to have the same values.
a = [1, 2, 3, 4]  # Point a at a new list, [1, 2, 3, 4]
b = a             # Point b at what a is pointing to
b is a            # => True, a and b refer to the same object
b == a            # => True, a's and b's objects are equal
b = [1, 2, 3, 4]  # Point b at a new list, [1, 2, 3, 4]
b is a            # => False, a and b do not refer to the same object
b == a            # => True, a's and b's objects are equal


id(256)
a=256
b=256
id(a), id(b), id(256)



row = [""]*3
board = [row]*3 # each of the elements board[0], board[1] and board[2] ...
                # ...is a reference to the same list referred by "row"
board[0][0]= "X"
board # ==> [['X', '', ''], ['X', '', ''], ['X', '', '']]
# to avoid this:
board_new = [[""]*3 for _ in range(3)] 
board_new[0][0] = "Y"
board_new   # ==> [['Y', '', ''], ['', '', ''], ['', '', '']]

#-----------------------------------------------------------------
# lists

# lists could contain different types of elements: strings, other lists, dictionaries, functions
[]
[1]
[1,2,3,4,5,6,7,8]
[1,"two",34.234,{"a","b"},(5,6)]

# list initialization: common one for working with large lists 
# whose size is known ahead of time
k=[None]*5
k

x = [12,23,34,13,24,35,46,57,68]
x[0], x[0:2], x[-1] # last element
x[1:-1]
x[-3:] # last 3 elements


46 in x
z = 13
z in x

# "if m in [1,2,3,4]:" is the same as "if m==1 or m==2 or m==3 or m==4:"

# Find the most frequent value in a list.
test = [1,2,3,2,4,5,6,5,5,7,8,8,7,5,4,5,3,4]
max(test,key=test.count)



testList = [1,2,3]
x,y,z=testList
x,y,z
print(x,y,z)


mixed_list = [False, 1.0, "some_string", 3, True, [],False]
integers_found_so_far=0
booleans_found_so_far=0
for i in mixed_list:
    if isinstance(i,int):
        integers_found_so_far += 1
    if isinstance(i, bool):
        booleans_found_so_far += 1
integers_found_so_far # Booleans are a subclass of int




# Adding/replace
    x[3:5]=["one","two","three"]
    [111,222,333]+x
    x[len(x):]=[222,333,444]
    x.append(1111)

    first = [1,2,3,4]
    second = [5,6,7]
    third = [8,9,10]
    first.append(second)
    first.extend(third)
    first

    x[8:9]=[]
    x.pop() # Remove from the end
    del x[2]
    x.remove("three")

    x.reverse() # this operation changes the initial list


# -----------------------------------------------------------------------------------
# DICTIONARIES
# Python´s hash tables

y = {}
y[0] = "Hallo"
y[1] = "Goodbye"

y["two"] = 2 # dictionary keys may be numbers, strings, etc (for lists - only integers)
y["pi"] = 3.14
y["two"]*y["pi"]

x = (1,2,3,4,5)
y = ("one","two","three","four","five")
dict(zip(x,y))

some_string = "snowboard"
some_dict = {}
for i, some_dict[i] in enumerate(some_string):
    pass
some_dict


# -----------------------------------------------------------------------------------
# LOOPS
item_list = [3, "string1", 23, 14.0, "string2", 49, 64,70]
for x in item_list: # Python’s for loop iterates over each of the items in a sequence (more of a foreach loop)
    if not isinstance(x,int):
        continue # If x isn’t an integer, the rest of this iteration is aborted by the continue statement
    if not x%7: # finds the first occurrence of an integer that’s divisible by 7
        print("Found an integer divisible by seven: %d" % x)
        break
        


for animal in ["dog", "cat", "mouse"]:
    print("{} is a mammal".format(animal))


for i in range(4):
    print(i)





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





#--------------------------------------------------------------
# CREATION OF PACKAGE

# Python modules are just ordinary Python files. 

import sys
sys.path
# When you ask Python to import a module, ...
# ... it starts with the first directory in sys.path ...
# ... and checks for an appropriate file. 
# If no match is found in the first directory it checks subsequent entries, in order, until a match is found 

# To create a normal module ...
# ... you simply createa Python source file in a directory contained in sys.path. 
# The process for creating packages is not much different. 




# You can write your own, and import them. The name of the module is the same as the name of the file.

'''File wo.py'''

def word_occur():           # new function, which counts occurrences of words in a file
    file_name = input("Enter the name of the file:") # Prompt user for the name of the file to use
    f = open(file_name, "r")
    word_list = f.read().split() # store words from the file in a list
    f.close()
    
    occurs_dict={}
    for word in word_list:
        occurs_dict[word] = occurs_dict.get(word,0) + 1 # increment the occurrences count for this word 
    print("File %s has %d words (%d are unique)", % (file_name, len(word_list), len(occurs_dict)))
    print(occur_dict)
    
if __name__=='__main__': # this allows the program to be run as a script by typing "python wo.py" at a command line
    word_occur
    
'''end of the file wo.py'''    
    
    
    
# If you place a file in one of the directories on the module search path, ...
# ... which can be found in sys.path, ...
# ... it can be imported like any of the built-in library modules by using the import statement:

import wo
wo.word_occur()

# Note that if you change the file wo.py on disk, ...
# ... import won’t bring your changes into the same interactive session. 
# You use the reload function from the imp library in this situation:

import imp
imp.reload(wo)



import re
dir(re) # which functions and attributes are defined in a module.

# There are also anonymous functions
(lambda x: x > 2)(3)                  # => True
(lambda x, y: x ** 2 + y ** 2)(2, 1)  # => 5



# To create a package, ...
# 1) create the package’s root directory. 
#        This root directory needs to be in some directory on sys.path
#        remember, this is how Python finds modules and packages for importing. 
# 2) In that root directory, you create a file called __init__.py. 
#        This file — which we’ll often call the package init file — is what makes the package a module. 
#        __init__.py can be(and often is) empty;
#        its presence alone suffices to establish the package.

# In Shell:
# mkdir Anaconda3/reader
# type Anaconda3/reader/__init__.py

import reader
type(reader) # => module, even though on our filesystem the name “reader” refers to a directory
reader.__file__

# ------------------------------------------------------------------
# try ... catch

try:
  df_input = df_input.loc[:,(met_model,model_params['variables'],slice(None))]
except KeyError as e:
  logger.warning(met_model+' not available : skipping it')
  continue


error_messages=[]
try:
    for i in pool_result:                                                                                                                                                      
        try:
            result = i.get()
        except Exception as err:
            error_messages+=(err)
except Exception as outer_err:
    error_messages+=(outer_err)


if len(error_messages)==0:
    logger.info('**************************************** ')
    logger.info('*         Succesfully finished         * ')
    logger.info('**************************************** ')
else:
    logger.error(error_messages)



# -------------------------------------------------------------------------------
#Push to G Drive

#https://medium.com/@annissouames99/how-to-upload-files-automatically-to-drive-with-python-ee19bb13dda                                                   
#pip install PyDrive is needed                                                                                                                           

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
g_login = GoogleAuth()
g_login.LocalWebserverAuth()
drive = GoogleDrive(g_login)

with open("results/file_to_be_pushed","r") as file:
    file_drive = drive.CreateFile({'title':os.path.basename(file.name) })
    file_drive.SetContentString(file.read())
    file_drive.Upload()

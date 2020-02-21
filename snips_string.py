# string series
# modify strings by converting them to list
# delete $ from string
# skip strings with //
# different ways of printing strings
# Convert raw string integer inputs to integers
# Concatenate long strings elegantly across line breaks in code
# extract US numbers from text
# clean spaces in strings
# getting rid of extra white spaces
# make translation of strings
# Convert "1k" to 1 000, "1m" to 1 000 000, etc.
# creates dictionary from string
# all possible letter combinations
# fuzzy matching
# the most often occuring names using collection.Counter
# Get vowels from a sentence


#-------------------------------------------------------------------
# string series
	Series(list('abcde'))
	random.choices(string.ascii_lowercase,k=5) # generates k random letters
	tm.makeStringIndex()


#-------------------------------------------------------------------
# modify strings by converting them to list

x ="Hello!"
x[-2] # like in the list ...
# x[2]="F" ... but strings cannot be modified
# x.append("v")
# => ...
#    1) turn the string into a list of characters, 
#    2) do whatever you want, 
#    3) and then turn the resulting list back into a string
text = "Hello world!"
wordList = list(text)
wordList[7:]=[] # removes everything after 7th char
wordList.reverse()
text="".join(wordList) # turn the list back into a string

#-------------------------------------------------------------------
# delete $ from string
df.state_bottle_retail.str.replace('$','') # 4.5*X ms: replaces the ‘$’ with a blank space for each item in the column
df.state_bottle_retail.apply(lambda x: x.replace('$','')) # 4*X ms: pandas ‘apply’ method, which is optimized to perform operations over a pandas column
df.state_bottle_retail.apply(lambda x: x.strip('$')) # 3*X ms: strip does one less operation: just takes out the ‘$.’
df.state_bottle_retail = [x.strip('$') for x in df.state_bottle_retail] # 2*X ms: list comprehension
df.state_bottle_retail = [x[1:] for x in df.state_bottle_retail] # X ms: built in [] slicing, [1:] slices each string from 2nd value till end


#----------------------------------------------------------------------------------------------------
# skip strings with //
string_from_file = """
// Author: ...
// License: ...
//
// Date: ...

Actual content...
"""

import itertools
for line in itertools.dropwhile(lambda line: line.startswith("//"), string_from_file.split("\n")):
	print(line)



#----------------------------------------------------------------------------------------------------
# printing strings

e=2.718
x = [1, "two", 3, 4.0, ["a", "b"], (5, 6)]
print ("The constant e is ", e, " and the list x is ", x) # converts everything to string
"{} can be {}".format("Strings", "interpolated")  # => "Strings can be interpolated"

print("the value of %s is: %.2f" % ("e", e)) # formatting capabilities similar to sprintf in c
num_dict = {'e':2.718,'pi':3.14159}
print("%(pi).2f - %(pi).4f - %(e).1f" % num_dict)

" ".join(["a","b","c","d"])

# --------------------------------------------------------------------
# Convert raw string integer inputs to integers

str_input = "1 2 3 4 5 6"
int_input = map(int, str_input.split())
print(list(int_input))

#--------------------------------------------------------------------------------------------------------
# Concatenate long strings elegantly across line breaks in code

my_long_text = ("We are no longer the knights who say Ni! "
                "We are now the knights who say ekki-ekki-"
                "ekki-p'tang-zoom-boing-z'nourrwringmm!")


#--------------------------------------------------------------------------------------------------------		
# extract US numbers from text
import re
numbers = ' 123.456.7889 (123)-456-7888 (425) 465-7523 456 123-7891 111 111.1111 (222)333-4444 666 777 8888 987-654-4321'
res = re.findall(r'\d{3}\)*?\-*?\s*?\.*?\d{3}\-*?\s*?\.*?\d{3}', numbers)		

#--------------------------------------------------------------------------------------------------------
# clean spaces in strings
		
import re
def clean(string): # cleans from empty spaces on start and on end
    # input is : "   Hello World    " 
    # output is "Hello World"
		
    first = 0
    for item in string:
        if item != ' ':
            first = string.index(item)
            break
    string = string[::-1]
    last = 0
    for item in string:
        if item != ' ':
            last = string.index(item)
            break
    return string[::-1][first:len(string)-last]
    
def clean2(string): # the same purpose as above
    nonempty = [string.index(item) for item in string if item != ' ']
    nonempty2 = [string[::-1].index(item) for item in string[::-1] if item != ' ']
    return string[nonempty[0]:len(string) -nonempty2[0]]

def clean3(string): # clean string from rebundant spaces
    try:
        for item in re.findall('[" "]{2,}',string):
            string = string.replace(item, ' ')
            if string[0] == ' ':
                string = string[1:]
            if string[-1] == ' ':
                string = string[:-1]
        return string
    except:
return string
		
#---------------------------------------------------------------------
# getting rid of extra white spaces
import string
string.whitespace # white spaces are not only spaces, but e.g. new line, tab, etc.

x="\n tha basketball playear allen iverson \t   "
x
x.strip() # any whitespaces in the beginning and end are removed
x.rstrip() # rstrip removes whitespace only at the right end

# The common use for these functions is to clean up strings...
# ... that have just been read in: e.g. reading lines from files ...
# ... as Python reads in an entire line, incl. the trailing newline, 
# "rstrip" is a convenient way to get rid of it.

#--------------------------------------------------------------------------------------------------------
# make translation of strings

x = "~x ^ (y % z)" 
table = x.maketrans("~^()","!&[]") # from package "string"
x.translate(table)

#--------------------------------------------------------------------------------------------------------
# Convert "1k" to 1 000, "1m" to 1 000 000, etc.

def resolve_value(value):
    if value is None:
        return None
    tens = dict(k=10e3, m=10e6, b=10e9, t=10e12)
    value = value.replace(',', '')
    match = re.match(r'(-?\d+\.?\d*)([kmbt]?)$', value, re.I)
    if not match:
        return None
    factor, exp = match.groups()
    if not exp:
        return float(factor)
     return int(float(factor)*tens[exp.lower()])

#--------------------------------------------------------------------------------------------------------
# creates dictionary from string
some_string = "snowboard"
some_dict = {}
for i, some_dict[i] in enumerate(some_string):
    pass
some_dict

#--------------------------------------------------------------------------------------------------------
# all possible letter combinations
"""
Given a digit string, return all possible letter
combinations that the number could represent.
Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
"""

def letter_combinations(digits):
    if digits == "":
        return []
    kmaps = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }
    ans = [""]
    for num in digits:
        tmp = []
        for an in ans:
            for char in kmaps[num]:
                tmp.append(an + char)
        ans = tmp
    return ans

#--------------------------------------------------------------------------------------------------------
# fuzzy matching with Levenshtein distance

import difflib
difflib.get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy'], n=2) # n specifies max num of matches to be returned
# returns ['apple', 'ape']


#--------------------------------------------------------------------------------------------------------
# the most often occuring names using collection.Counter
from collections import Counter

cheese = ["gouda", "brie", "feta", "cream cheese", "feta", "cheddar",
          "parmesan", "parmesan", "cheddar", "mozzarella", "cheddar", "gouda",
          "parmesan", "camembert", "emmental", "camembert", "parmesan"]

cheese_count = Counter(cheese) # Counter is just a dictionary that maps items to number of occurrences
# use update(more_words) method to easily add more elements to counter

print(cheese_count.most_common(3))
# Prints: [('parmesan', 4), ('cheddar', 3), ('gouda', 2)]


# ---------------------------------------------------------------------------------------------------
# Get vowels from a sentence
sentence = 'the rocket came back from mars'
vowels = [i for i in sentence if i in 'aeiou'] # list
unique_vowels = {i for i in sentence if i in 'aeiou'} # set




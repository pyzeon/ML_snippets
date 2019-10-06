
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



x="tha basketball playear allen iverson"
x
x.replace("allen iverson","kobe bryant")
x.split()
x.split("e")




e=2.718
x = [1, "two", 3, 4.0, ["a", "b"], (5, 6)]
print ("The constant e is ", e, " and the list x is ", x) # converts everything to string
"{} can be {}".format("Strings", "interpolated")  # => "Strings can be interpolated"

print("the value of %s is: %.2f" % ("e", e)) # formatting capabilities similar to sprintf in c
num_dict = {'e':2.718,'pi':3.14159}
print("%(pi).2f - %(pi).4f - %(e).1f" % num_dict)





" ".join(["a","b","c","d"])









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
def clean(string): # cleans string from empty spaces on the start and on the end
    # input is : "   Hello World    " and output is "Hello World"
		
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
		
# getting rid of extra white spaces
import string
string.whitespace # white spaces are not only spaces, but e.g. new line, tab, etc.

x="\n tha basketball playear allen iverson \t   "
x
x.strip() # any whitespaces in the beginning and end are removed
x.rstrip() # rstrip removes whitespace only at the right end of the original string

# The most common use for these functions is as a quick way to clean up strings...
# ... that have just been read in: e.g. reading lines from files ...
# ... because Python always reads in an entire line, including the trailing newline, 
# "rstrip" is a convenient way to get rid of it.

#--------------------------------------------------------------------------------------------------------

# from package "string"
x = "~x ^ (y % z)" 
table = x.maketrans("~^()","!&[]") 
x.translate(table)

#--------------------------------------------------------------------------------------------------------

import re

def resolve_value(value):
    """
    Convert "1k" to 1 000, "1m" to 1 000 000, etc.
    """
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

some_string = "snowboard"
some_dict = {}
for i, some_dict[i] in enumerate(some_string):
    pass
some_dict

#--------------------------------------------------------------------------------------------------------


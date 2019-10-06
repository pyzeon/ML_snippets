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

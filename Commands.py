
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

       with zipfile.ZipFile(self.fname) as zf:
            with zf.open(zf.namelist()[0]) as infile:
                header = infile.readline()
                datestr, record_count = header.split(b':')
                self.month = int(datestr[2:4])
                self.day = int(datestr[4:6])
                self.year = int(datestr[6:10])
                utc_base_time = datetime(self.year, self.month, self.day)
                self.base_time = timezone('US/Eastern').\
                                                localize(utc_base_time).\
                                                timestamp()

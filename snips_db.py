# names of csvs
# gets the list of tickers in the directory
# read txt
# skip lines when reading text
# download the zip file with many txts and move the data to 1 csv
# divide text (csv or ...) to small files with defined number of lines
# Load lines from csv file
# check existing and create txts for every ticker
# update csv from the latest date to today
# create folder with curr date
# check latest data in file
# read all needed csvs from zip		

# from MongoDB to excel
# get last date of data from Postgres DB


#--------------------------------------------------------------------------------------------------------
# names of csvs
if fname[-4:] == '.csv':
	fname = fname[:len(fname)-4]
	    

#--------------------------------------------------------------------------------------------------------
# gets the list of tickers in the directory

def get_list_tickers_in_dir(directory=None):
    start_dir = os.getcwd()
    if directory == None:
        directory = start_dir
        
    listdir=os.listdir(directory)
    TCKRS=[]
    for ls in listdir:
        if not ls.endswith('.csv'):
            continue        
        check = re.match("[A-Z]*?\.[A-Z].",ls)
        if check==None:
            continue
        TCKRS.append(check.group()[:-1])
    
    TCKRS = list(set(TCKRS)) # remove duplicates
    os.chdir(start_dir)
    return (TCKRS, listdir) 

#--------------------------------------------------------------------------------------------------------
# read txt

filename = "myfile.txt"
with open(filename, "r") as f: # automaticall close the file in the end
    for line in f:
        print(f)

# This above is equivalent to this:
filename = "myfile.txt"
try: 
    f = open(filename, "r")
    for line in f:
        print(f)
except Exception as e:
    raise e
finally:
    f.close()

#--------------------------------------------------------------------------------------------------------
# skip lines when reading text
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



#--------------------------------------------------------------------------------------------------------
# from MongoDB to excel
# Connectio URI can be in shape mongodb://<username>:<password>@<ip>:<port>/<authenticationDatabase>')
client = pymongo.MongoClient('mongodb://localhost')

def export_to_excel(name, collection, database):
    data = list(client[database][collection].find({},{'_id':0}))
    df =  pd.DataFrame(data)
    df.to_excel('{}.xlsx'.format(name)') #writer, sheet_name='Sheet1')

#--------------------------------------------------------------------------------------------------------
# get last date of data from Postgres DB
def fetch_last_day_mth(year_, conn):
    cur = conn.cursor() # conn: a Postgres DB connection object
    SQL =   """
            SELECT MAX(date_part('day', date_price)) FROM daily_data
            WHERE date_price BETWEEN '%s-12-01' AND '%s-12-31'
            """
    cur.execute(SQL, [year_,year_])        
    data = cur.fetchall()
    cur.close()
    last_day = int(data[0][0])
    return last_day

		
#--------------------------------------------------------------------------------------------------------
# download the zip file with many txts and move the data to 1 csv
import requests
url = "https://www.ssa.gov/oact/babynames/names.zip"
with requests.get(url) as response:
    with open("names.zip", "wb") as temp_file:
        temp_file.write(response.content)

data_list = [["year", "name", "gender", "count"]] # 2-dimensional Array (list of lists)

with ZipFile("names.zip") as temp_zip: # open the zip file into memory
    for file_name in temp_zip.namelist(): # Then we read the file list.
        if ".txt" in file_name: # We will only process .txt files.
            with temp_zip.open(file_name) as temp_file: # read the current file from the zip file.
                # The file is opened as binary, we decode it using utf-8 so it can be manipulated as a string.
                for line in temp_file.read().decode("utf-8").splitlines():
                    line_chunks = line.split(",")
                    year = file_name[3:7]
                    name = line_chunks[0]
                    gender = line_chunks[1]
                    count = line_chunks[2]

                    data_list.append([year, name, gender, count])

csv.writer(open("data.csv", "w", newline="", # We save the data list into a csv file.
                encoding="utf-8")).writerows(data_list)
                # I prefer to use writerows() instead of writerow() ...
                # ...since it is faster as it does it in bulk instead of one row at a time.

#--------------------------------------------------------------------------------------------------------
# divide text (csv or ...) to small files with defined number of lines

def splitter(name, parts = 100000):
    # make dir for files
    if not os.path.exists(name.split('.')[0]): 
        os.makedirs(name.split('.')[0])
    f = open(name, 'r', errors = 'ignore')
    lines = f.readlines()
    f.close()
    i = 0
    while i < len(lines):
        for item in lines[i:i+parts]:
            f2 = open(name.split('.')[0]+ '/'name.split('.')[0]+ str(i)+'.txt', 'a+', errors = 'ignore') 
            f2.write(item)
            f2.close()
    i += parts

#-----------------------------------------------------------------------------------------------------------------------
# Load lines from csv file
		
def read_line_from_file(filename):    
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            lines.append(line.rstrip())
    if len(lines) > 0:
        lines = lines[1:]
    return lines

#-----------------------------------------------------------------------------------------------------------------------	
# check existing and create txts for every ticker		
if os.path.exists('{}'.format(path)):
	response = input('A database with that path already exists. Are you sure you want to proceed? [Y/N] ')
	if response == 'Y':
		for item in os.listdir('{}/trades/'.format(path)):
			os.remove('{}/trades/{}'.format(path, item))
		os.rmdir('{}/trades/'.format(path))
		for item in os.listdir('{}'.format(path)):
			os.remove('{}/{}'.format(path, item))
		os.rmdir('{}'.format(path))
print('Creating a new database in directory: {}/'.format(path))
self.trades_path = '{}/trades/'.format(path)
os.makedirs(path)
os.makedirs(self.trades_path)
for name in names:
	with open(self.trades_path + 'trades_{}.txt'.format(name), 'w') as trades_file:
		trades_file.write('sec,nano,name,side,shares,price\n')
					
#-----------------------------------------------------------------------------------------------------------------------	
# update csv from the latest date to today

DATE_FORMAT = "%Y-%m-%d"

def write_to_file(fn, f):
    if os.path.isfile(fn):
        f1 = open(fn, "r")
        last_line = f1.readlines()[-1]
        f1.close()
        last = last_line.split(",")
        date = (datetime.datetime.strptime(last[0], DATE_FORMAT)).strftime(DATE_FORMAT)
        today = datetime.datetime.now().strftime(DATE_FORMAT)
        if date != today:
            with open(fn, 'a') as outFile:
                f.tail(1).to_csv(outFile, header=False)
    else:
        print("new file")
        f.to_csv(fn)

#-----------------------------------------------------------------------------------------------------------------------	

# create folder with curr date in curr directory
date = datetime.date.today()
s_dir = os.getcwd() + '/' + date.strftime('%Y-%m-%d')
if not os.path.exists(s_dir):
    os.makedirs(s_dir)
		
#-----------------------------------------------------------------------------------------------------------------------			
# check latest data in file
if os.path.isfile(path):        
	df = pd.read_csv(path,index_col=0,header=0) 
	latest_date=df[df.index==max(df.index)]['DATE']
	latest= pd.datetime.strptime(latest_date[0],'%Y-%m-%d')
	ndays = pd.datetime.today().date()-latest.date()
	return str(ndays.days) + 'd'		
		
#-----------------------------------------------------------------------------------------------------------------------			
# read all needed csvs from zip		
fracfocus_url='http://fracfocusdata.org/digitaldownload/fracfocuscsv.zip'
request = requests.get(fracfocus_url)
zip_file = zipfile.ZipFile(io.BytesIO(request.content)) #generates a ZipFile object
list_of_file_names = zip_file.namelist() #list of file names in the zip file
list_to_append_to=[]
for file_name in list_of_file_names:
    if ((file_name.endswith('.csv')) & (key_word in file_name)):
        list_to_append_to.append(file_name)
list_of_dfs=[pd.read_csv(zip_file.open(x), low_memory=False) for x in list_to_append_to]

#-----------------------------------------------------------------------------------------------------------------------			
# check directories	
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
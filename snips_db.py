#read in latest_dates
if not(os.path.isfile(directory+'\\latest_dates\\latest_dates.csv')):
    print 'No latest_date.csv file found'
    print 'program terminated'
    return        

#--------------------------------------------------------------------------------------------------------
# gets the list of tickers in the directory

import os
import re

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
def get_csv_file_list(TCKR,listdir, directory=None):
    start_dir = os.getcwd() #save start dir
    if directory==None:
        directory = start_dir    
    os.chdir(directory)
    
    if listdir==0: #enables not entering valid listdir
        listdir = os.listdir(directory)
        
    ls = str(listdir)
        
    #search for single run files    
    check = re.search('\''+TCKR+'\.[0-9]*\.csv\', ',ls)
    if check == None:
        print 'Error: ticker='+TCKR+' single date csv not found in:'
        print directory
        return 'no tickers'
    files=[]
    while check !=None:
        s = str(check.group())
        s=s.replace("'","")
        s=s.replace(", ","")
        files.append(s)
        ls=ls.replace(check.group(),'xxxxx, ',1)  #replace 1st instance of filename wit xxxxx          
        check = re.search("\'"+TCKR+"\.[0-9]*\.csv\', ",ls)     
    
    os.chdir(start_dir)
    return files

#--------------------------------------------------------------------------------------------------------



f = open("myfile", "w") # creation of file object, in the current working dir in write mode
f.write("This is the first test line in the fiele. \n")
f.write ("And this is the second and probably the last line. \n")
f.close()


print("a","b","c", file=open("testfile.txt","w"))


f=open("myfile","r")
line1 = f.readline()
line2 = f.readline()
f.close()
print(line1, line2)

#--------------------------------------------------------------------------------------------------------

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

'''
Short function using Pandas to export data from MongoDB to excel
'''
import pandas as pd
from pymongo import MongoClient

# Connectio URI can be in shape mongodb://<username>:<password>@<ip>:<port>/<authenticationDatabase>')
client = MongoClient('mongodb://localhost')

def export_to_excel(name, collection, database):
    '''
    save collection from MongoDB as .xlsx file, name of file is argument of function 
    collection <string> is name of collection 
    database <string> is name of database
    '''
    data = list(client[database][collection].find({},{'_id':0}))
    df =  pd.DataFrame(data)
    #writer = pd.ExcelWriter('{}.xlsx'.format(name), engine='xlsxwriter')
    df.to_excel('{}.xlsx'.format(name)') #writer, sheet_name='Sheet1')
    #writer.save()


#--------------------------------------------------------------------------------------------------------

def fetch_last_day_mth(year_, conn):
    """
    return date of the last day of data we have for a given year in our Postgres DB. 
    conn: a Postgres DB connection object
    """  
    cur = conn.cursor()
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
# download the zip file and saved it to our computer
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

'''divide text (csv or ...) to small files with defined number of lines'''
import os

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

def read_line_from_file(filename):
    """Load lines from csv file"""
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            lines.append(line.rstrip())
    if len(lines) > 0:
        lines = lines[1:]
    return lines

#-----------------------------------------------------------------------------------------------------------------------	

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

def create_directories(self):
        main_directory = "PairsResults"+self.params
        
        if not os.path.exists(main_directory):
            os.makedirs(main_directory)
        if not os.path.exists(self.directory_pair):
            os.makedirs(self.directory_pair)		

#-----------------------------------------------------------------------------------------------------------------------	

import pandas as pd
import os

DATE_FORMAT = "%Y-%m-%d"

def file_exists(fn):
    exists = os.path.isfile(fn)
    if exists:
        return 1
    else:
        return 0

def write_to_file(exists, fn, f):
    if exists:
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

import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(parent_dir, 'data', 'sr')
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

#-----------------------------------------------------------------------------------------------------------------------	
import os
import datetime as dt

if date == None:
    date = dt.date.today()

s_dir = os.getcwd() + '/' + date.strftime('%Y-%m-%d')
if not os.path.exists(s_dir):
    os.makedirs(s_dir)

#-----------------------------------------------------------------------------------------------------------------------	

import os
os.getcwd()
print(os.getcwd())
		
		
#-----------------------------------------------------------------------------------------------------------------------			
def get_google_days(TCKR):
    path='D:\\...\\'+TCKR+'.combined.csv'
    if os.path.isfile(path):        
        df = pd.read_csv(path,index_col=0,header=0) 
        latest_date=df[df.index==max(df.index)]['DATE']
        latest= pd.datetime.strptime(latest_date[0],'%Y-%m-%d')
        ndays = pd.datetime.today().date()-latest.date()
        return str(ndays.days) + 'd'
    else:
        return '100d'		
		
		
#-----------------------------------------------------------------------------------------------------------------------			

fracfocus_url='http://fracfocusdata.org/digitaldownload/fracfocuscsv.zip'
request = requests.get(fracfocus_url)
zip_file = zipfile.ZipFile(io.BytesIO(request.content)) #generates a ZipFile object
list_of_file_names = zip_file.namelist() #list of file names in the zip file
list_to_append_to=[]
for file_name in list_of_file_names:
    if ((file_name.endswith('.csv')) & (key_word in file_name)):
        list_to_append_to.append(file_name)
list_of_dfs=[pd.read_csv(zip_file.open(x), low_memory=False) 
                for x in list_to_append_to]

#-----------------------------------------------------------------------------------------------------------------------			


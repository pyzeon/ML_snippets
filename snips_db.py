

# read txt with "with" and open()
    # filename = "myfile.txt"
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


    with open('foo.txt', 'rt') as file: # 'rt' - open for reading (text)
        data = file.read()
        # `data` is a string with all the text in `foo.txt`

    with open(filename, 'rt') as file:
        for line in file:
            if line == '\n':    # Skip blank lines
                continue
            # More statements
            ...

    with open(filename) as f:
        for lineno, line in enumerate(f, start=1):
        ...




    [line.strip() for line in open("readFile.py")] # here Python closes the file automatically




    with open('outfile', 'wt') as out:  # 'wt' - open for writing (text)
        out.write('Hello World\n')

    with open('outfile', 'wt') as out:
        print('Hello World', file=out)
        ...

    records = []  # Initial empty list

    with open('Data/portfolio.csv', 'rt') as f:
        next(f) # Skip header
        for line in f:
            row = line.split(',')
            records.append((row[0], int(row[1]), float(row[2])))

    import csv
    def portfolio_cost(filename):
        '''Computes the total cost (shares*price) of a portfolio file'''
        total_cost = 0.0

        with open(filename, 'rt') as f:
            rows = csv.reader(f)
            headers = next(rows)
            for row in rows:
                nshares = int(row[1])
                price = float(row[2])
                total_cost += nshares * price
        return total_cost


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


    def ls(dir="."):
        """ List files in current directory or directory passed as parameter """
        import os # Inside the method to avoid it being added to global namespace
        return [os.path.abspath(dir), os.listdir(dir)]


#Create list of avialable FX pairs
    import os, glob
    import pandas as pd
    directory = 'D:\\Data\\minute_data\\histdata_2000_2019\\'
    file_list = glob.glob(directory + '*.zip') # all files in directory
    hist_data = pd.DataFrame(file_list)
    hist_data['curr_pair'] = hist_data[0].apply(lambda x: x[10:16])
    hist_data['year'] = hist_data[0].apply(lambda x: x[-8:-4])


# Read minute data from different tickres in one dataframe
    import os, glob
    import pandas as pd

    sp500_directory = 'D:\\Data\\minute_data\\spx-1m-adj-csv\\'
    combined = pd.concat([pd.read_csv(f, sep=',', decimal=".", 
                                        usecols=[0,1], 
                                        names=("Day", "Time"),
                                        nrows=1,
                                        # skiprows=range(2,count-1), 
                                        header=None).
                            assign(filename = f) 
                            for f in glob.glob(sp500_directory + '*.txt')])
    combined['Symbol'] = [x.split('.')[0] for x in os.listdir(sp500_directory) if x.endswith(".txt")] # names of txt files in the directory
    combined.to_csv(sp500_directory + 'SP500_TKRS.csv', index=False)



    from collections import deque
    def read_fst_lst(filename):
        count=len(open(filename).readlines()) 
        pd.read_csv(filename, sep=',', decimal=".", usecols=[0,1], skiprows=range(2,count-1), header=None).assign(filename = filename)
    combined = pd.concat([read_fst_lst(f) for f in glob.glob(sp500_directory + '*.txt')])


# Get info from different places in line
    for line in listdir:
        line_chunks = line.split("_")
        year = line[10:16]
        name = line_chunks[0]
        gender = line_chunks[1]
        count = line_chunks[2]

        data_list.append([year, name, gender, count])


# Reading quotes to dataframe and analysing it
    import os, glob
    import pandas as pd

    directory = 'D:\\Data\\minute_data\\'
    file_list = glob.glob(directory + '*.txt') # all files in directory
    li=[x.split('.')[0] for x in os.listdir(directory) if x.endswith(".txt")]
    hist_data = pd.DataFrame(file_list)
    hist_data['ticker'] = hist_data[0].str.split("\")[-1]

    read_file = pd.read_csv(file_list[0], sep='\t', decimal=",")
    read_file['Date'] = pd.to_datetime(read_file['Date']+' '+read_file['Time'] # merge 2 columns
                                        , format='%d.%m.%Y %H:%M') # adding format makes converting much faster
    read_file=read_file.drop(read_file.columns[[1,-1]],axis=1) # delete not needed columns
                        .set_index('Date')



    dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')
    AAPL = pd.read_csv("D:\\Data\\minute_data\\AAPL.txt", sep='\t', decimal=",", 
                        parse_dates={'datetime': ['Date', 'Time']}, date_parser=dateparse)
    AAPL.set_index('datetime',inplace=True) # "inplace" make the changes in the existing df
    AAPL=AAPL.drop(AAPL.columns[[-1]],axis=1) # delete last column



    read_file.dtypes
    read_file.resample("Y").count()
    read_file[read_file.isna().any(axis=1)].count()
    read_file[read_file.isnull().any(axis=1)].count()


    extract=[read_file[c].nlargest(2).iloc for c in read_file]
    read_file.loc[extract]


    read_file.nlargest(3,'Volume',keep='all')

    read_file[read_file.nlargest(3,x,keep='all') for x in read_file]

    read_file.min()


    read_file.mask((read_file - read_file.mean()).abs() > 4 * read_file.std())
    read_file[read_file.apply(lambda x :(x-x.mean()).abs()>(3*x.std()) ).all(1)]


# Loading sample of big csv:
    df = pd.read_csv("/.../US_Accidents_Dec19.csv", 
                    skiprows = lambda x: x>0 # x > 0 makes sure that the headers is not skipped 
                                        and np.random.rand() > 0.01) # returns True 99% of the time, thus skipping 99% of the time


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


# Merge all csv files of the same struc in the same folder
    import glob
    import pandas as pd
    from time import strftime

    def folder_csv_merge(file_prefix, folder_path='', memory='no'):
        if folder_path == '':
            folder_path = input('Please enter the path where the CSV files are:\n')
        folder_path = folder_path.replace("\\","/")
        if folder_path[:-1] != "/":
            folder_path = folder_path + "/"

        file_list = glob.glob(folder_path + '*.csv')

        combined = pd.concat( [ pd.read_csv(f) for f in file_list ] )
        if memory == 'no':
            combined.to_csv(folder_path + 
                            'combined_{}_{}.csv'.format(file_prefix, 
                                                strftime("%Y%m%d-%H%M%S")), 
                            index=False)
        else:
            return combined
        print('done')


# read all csvs, merge and reduce the size of big file from 30GB to 10GB   
    wdir = "C:/bigdata/pums/2014-2018/pop"
    os.chdir(wdir)
    all_files = glob.glob("*.csv")     
    pop_list = (pd.read_csv(f) for f in all_files)
    popr = pd.concat(pop_list, ignore_index=True)

    def mkdowncast(df): # reducing the size of big file from 30GB to 10GB   
        for c in enumerate(df.dtypes) : 
            if c[1] in ["int32","int64"] : 
                df[df.columns[c[0]]] = pd.to_numeric(df[df.columns[c[0]]], downcast='integer')
        for c in enumerate(df.dtypes) : 
            if c[1] in ["float64","float32"] : 
                df[df.columns[c[0]]] = pd.to_numeric(df[df.columns[c[0]]], downcast='float')
        return(df)

    poprd = mkdowncast(popr.copy())

    # Save memory of a dataframe by converting to smaller datatypes
    df = pd.read_csv("../input/titanic/train.csv", usecols = ["Pclass", "Sex", "Parch", "Cabin"])
    df.memory_usage(deep = True) # let's see how much our df occupies in memory

    # convert to smaller datatypes
    df = df.astype({"Pclass":"int8",
                    "Sex":"category", 
                    "Parch": "Sparse[int]", # most values are 0
                    "Cabin":"Sparse[str]"}) # most values are NaN
    df.memory_usage(deep = True)


#collect all cleaned csv files
    def collect_cleaned_bones(file_path):
        lst = []
        
        #get all subdirectories
        dirs = walk(file_path)
        i = 0
        
        for dir in dirs:
            print "Checking "+dir[0]
            filenames = [f for f in listdir(dir[0]) if isfile(join(dir[0], f))]
            for filename_ in filenames:
                if filename_.lower().endswith('_tmp.csv'):
                    lst.append(dir[0]+"\\"+filename_)
                    print "Collected %s" %i
                    i += 1    
        #return unique list
        return list(set(lst))


# Detect and Replace Missing Values in CSV
    import pandas as pd
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    messagebox.showinfo("Missing Data Imputer", "Click OK to Choose your File.")
    root.withdraw()
    file_path = filedialog.askopenfilename()
    df=pd.DataFrame(pd.read_csv(file_path))
    if df.isnull().values.any():
        affected_cols = [col for col in df.columns if df[col].isnull().any()]
        affected_rows = df.isnull().sum()
        missing_list = []
        for each_col in affected_cols:
        missing_list.append(each_col)
    for each in missing_list:
        df[each] = df[each].interpolate()


# Convert quotes from csv to dictionary
    # Source: qstrader/price_handler/historic_csv_tick.py

    def _open_ticker_price_csv(self, ticker):
        ticker_path = os.path.join(self.csv_dir, "%s.csv" % ticker)
        self.tickers_data[ticker] = pd.io.parsers.read_csv(
            ticker_path, header=0, parse_dates=True,
            dayfirst=True, index_col=1,
            names=("Ticker", "Time", "Bid", "Ask")
        )


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


    def unzip(file_path, subdir):
        import zipfile
        
        #walk through folder and unzip all
        i = 0
        d = 0
        
        #if archives were inside archives
        if subdir:
            dirs = walk(file_path)
            
            for dir in dirs:
                if d > 0:
                    print "Checking "+dir[0]
                    rename_elements_in_archives(dir[0], True)
                    filenames = [f for f in listdir(dir[0]) if isfile(join(dir[0], f))]
                    for filename in filenames:
                        if "tmp" in filename:
                            try:
                                print i
                                print "Unzipping "+dir[0]+"\\"+filename
                                with zipfile.ZipFile(dir[0]+"\\"+filename, "r") as zipped:
                                    zipped.extractall(dir[0]+"\\")
                                i += 1
                            except Exception as e:
                                print e
                d += 1
        #normal way
        else:
            filenames = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        
            for filename in filenames:
                if "tmp" in filename:
                    try:
                        print "Unzipping %s" %i
                        print filename
                        with zipfile.ZipFile(file_path+filename, "r") as zipped:
                            zipped.extractall(file_path)
                            i += 1
                    except Exception as e:
                        print e


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
					

# check latest data in file and update csv from the latest date to today

    if os.path.isfile(path):        
        df = pd.read_csv(path,index_col=0,header=0) 
        latest_date=df[df.index==max(df.index)]['DATE']
        latest= pd.datetime.strptime(latest_date[0],'%Y-%m-%d')
        ndays = pd.datetime.today().date()-latest.date()
        return str(ndays.days) + 'd'		

    DATE_FORMAT = "%Y-%m-%d"
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

    # Get last n lines of a file:
    def tail(filename, n=10):
        with open(filename) as f:
            return collections.deque(f, n)

    # get last date from Postgres DB
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


# Keep track of where your data is coming when you are using multiple sources

    # let's generate some fake data
    df1 = generate_sample_data()
    df2 = generate_sample_data()
    df3 = generate_sample_data()
    df1.to_csv("trick78data1.csv")
    df2.to_csv("trick78data2.csv")
    df3.to_csv("trick78data3.csv")

    # Step 1 generate list with the file name
    lf = []
    for _,_, files in os.walk("/kaggle/working/"):
        for f in files:
            if "trick78" in f:
                lf.append(f)
                
    lf

    # You can use this on your local machine
    #from glob import glob
    #files = glob("trick78.csv")

    # Step 2: assing create a new column named filename and the value is file
    # Other than this we are just concatinating the different dataframes
    df = pd.concat((pd.read_csv(file).assign(filename = file) for file in lf), ignore_index = True)
    df.sample(10)


# Utility to output an n-level nested dictionary as a CSV

    import csv
    import os

    def flatten_dict(
            data, # Dictionary to be parsed
            parent_key='',
            sep='_'):
        items = []
        for key, value in data.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, dict):
                items.extend(flatten_dict(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                for idx, val in enumerate(value):
                    temp_key = f'{new_key}_{idx}'
                    items.extend(flatten_dict(
                        val,
                        temp_key,
                        sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)
    # end of flatten_dict


# Convert a dictionary to a CSV
    def dict_to_csv(
            data, # Dictionary to be converted
            filename='test'):
        noext_filename = os.path.splitext(filename)[0]
        flattened_data = flatten_dict(data)
        with open(f'{noext_filename}.csv', 'w') as f:
            w = csv.DictWriter(f, flattened_data.keys())
            w.writeheader()
            w.writerow(flattened_data)


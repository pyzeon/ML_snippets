segments.seg_length.hist(bins=500)
segments.seg_length.apply(np.log).hist(bins=500)


# CREATE SERIES OR DATAFRAME (many series) --------------------------------------------------------------------

# Series creation
	Series([1,2,3,4]), Series([3,4,5,6,7],index=['a','b','c','d','e']), Series([3]*5), np.zeros(10)
	Series(np.arange(4,9)) # using the numpy function
	Series(np.linspace(0,9,5)) # allows to specify the number of values to be created btw boundaries

	pd.date_range('2016-08-01','2017-08-01')
	dates = pd.date_range('2016-08-01','2017-08-01', freq='M')
	idx = pd.date_range("2018-1-1",periods=20,freq="H")
	ts = pd.Series(range(len(idx)),index=idx)
	ts.resample("2H").mean()

	pd.Series(range(10),index=pd.date_range("2000",freq="D",periods=10))

# dummy datasets with dates
	import pandas.util.testing as tm
	tm.N, tm.K = 5,3
	tm.makeTimedeltaIndex(), tm.makeTimeSeries(), tm.makePeriodSeries()
	tm.makeDateIndex(), tm.makePeriodIndex(), tm.makeObjectSeries()

# random series
	Series(np.random.normal(size=5))
	np.random.randint(50,101,len(dates))

	tm.makeFloatSeries(), tm.makeBoolIndex(), tm.makeCategoricalIndex()
	tm.makeCustomIndex(nentries=4,nlevels=2), tm.makeFloatIndex(), tm.makeIntIndex()
	tm.makeMultiIndex(), tm.makeRangeIndex(), tm.makeIntervalIndex()

# string series
	Series(list('abcde'))
	random.choices(string.ascii_lowercase,k=5) # generates k random letters
	tm.makeStringIndex()

# Create dataframes 
	dates=pd.date_range('2016-08-01','2017-08-01')
	s = Series(np.random.randint(50,60,size=len(dates)))
	temp_df=pd.DataFrame({'Hurra':dates, 'Pinguin':s})

	from itertools import product
	datecols = ['year', 'month', 'day']
	df = pd.DataFrame(list(product([2016,2017],[1,2],[1,2,3])),columns = datecols)
	df['data']=np.random.randn(len(df))
	df.index = pd.to_datetime(df[datecols])

	
        open_orders = []
        for order in orders:
            if order['state'] == 'queued':
                open_orders.append(order)
		
	queried_symbols = [ fundamentals['symbol'] for fundamentals in queried_fundamentals ]
	
	
	
# quickly create a dataframe for testing
	import pandas.util.testing as tm
	tm.N, tm.K = 5,3
	tm.makeDataFrame(), tm.makeMixedDataFrame(), tm.makeTimeDataFrame(freq="W")

np.random.random((3,3,3)) # Create a 3x3x3 array with random values	



# Importing data -----------------------------------------------------------------------
SPX500=pd.read_csv("D:\\Data\\tick_data\\tick_data_zorro\\SPX500_2015.csv")
SPY_TICK=pd.read_csv("D:\\Data\\tick_data\\SPY_TICK_TRADE.csv")
GBPCAD=pd.read_csv("D:\\Data\\minute_data\\GBPCAD_2017.csv", header=False, parse_dates=['Date'])
NQ100=pd.read_csv("http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx?render=download",
                 usecols=[0,1,2,5],
                 index_col='Symbol',
                 skipinitialspace=True)

pandas.read_csv('http://www.nasdaq.com/investing/etfs/etf-finder-results.aspx?download=Yes')['Symbol'].values

calls_df, = pd.read_html("http://apps.sandiego.gov/sdfiredispatch/", header=0, parse_dates=["Call Date"])

#errors='coerce' means that we force the conversation.
#Values that can not be converted are set to NaN ("Not a Number")
data['educ'] = pd.to_numeric(data['educ'],errors='coerce') 





spyderdat = pd.read_csv("/home/curtis/Downloads/HistoricalQuotes.csv")    # Obviously specific to my system; set to                                                                        # location on your machine
spyderdat = pd.DataFrame(spyderdat.loc[:, ["open", "high", "low", "close", "close"]]
                        .iloc[1:].as_matrix(),
                         index=pd.DatetimeIndex(spyderdat.iloc[1:, 0]),
                         columns=["Open", "High", "Low", "Close", "Adj Close"])
            .sort_index()
spyder = spyderdat.loc[start:end]
stocks = stocks.join(spyder.loc[:, "Adj Close"])
                .rename(columns={"Adj Close": "SPY"})




	
#----------------------------------------------------------------------------------------------------------
# Describe and actions with series ------------------------------------------------------------------------

s = Series(np.random.randint(10,100,size =30))
s[3], s[[1,3]], s[3:17:4] # step 4
s.head(), s[:5], s[:-2] # all but the last 2

bob = Series(np.arange(3,30,3))
bob >15
bob[(bob>15) & (bob<25)]


line = Series(np.random.randint(1,200,size=1000))
line.sample(n=3)
line.sample(frac=0.05) #selects 5% of data

np.nonzero([1,2,0,0,4,0]) # find indices of non-zero elements

s= Series(np.random.randint(50,60,size=20))
s.values, s.index
len(s), s.size
s.unique(), s.value_counts(), s.nunique()# number of unique values
s.mean(), s.describe(), s.idxmax()
s.rank()

# Find Index of Min Element
lst = [40, 10, 20, 30]
min(range(len(lst)), key=lst.__getitem__)



# If array, Min, Max value was given, it returns array that contains
# values of given array which was larger than Min, and lower than Max.
def limit(arr, min_lim=None, max_lim=None):
    min_check = lambda val: True if min_lim is None else (min_lim <= val)
    max_check = lambda val: True if max_lim is None else (val <= max_lim)    
    return [val for val in arr if min_check(val) and max_check(val)]





bob = Series(np.arange(3,30,3))
(bob>10).any() # bool
(bob>2).all() # bool
(bob>15).sum()
bob.isnull().sum()


# find the closest value (to a given scalar)
	Z = np.arange(100)
	v = np.random.uniform(0,100)
	index = (np.abs(Z-v)).argmin()
	print(Z[index])

# find common values between two arrays
	Z1 = np.random.randint(0,10,10)
	Z2 = np.random.randint(0,10,10)
	print(np.intersect1d(Z1,Z2))


	dctA = {'a': 1, 'b': 2, 'c': 3}
	dctB = {'b': 4, 'c': 3, 'd': 6}
	"""loop over dicts that share (some) keys"""
	for ky in dctA.keys() & dctB.keys():
    	print(ky)
	"""loop over dicts that share (some) keys and values"""
	for item in dctA.items() & dctB.items():
    	print(item)


# Check if arrays are equal
	A = np.random.randint(0,2,5), B = np.random.randint(0,2,5)
	equal = np.allclose(A,B) # Assuming identical shape of the arrays and a tolerance for the comparison of values 
	equal = np.array_equal(A,B) # Checking both the shape and the element values, no tolerance (values have to be exactly equal)


# if you suggest 20%, it will neglect the best 10% of values
# and the worst 10% of values.
def trimmean(arr, per):
    ratio = per/200
    # /100 for easy calculation by *, and /2 for easy adaption to best and worst parts.
    cal_sum = 0
    # sum value to be calculated to trimmean.
    arr.sort()
    neg_val = int(len(arr)*ratio)
    arr = arr[neg_val:len(arr)-neg_val]
    for i in arr:
        cal_sum += i
    return cal_sum/len(arr)



# most_frequent_value
def top_1(arr):
    values = {}
    result = []
    f_val = 0

    for i in arr:
        if i in values:
            values[i] += 1
        else:
            values[i] = 1

    f_val = max(values.values())
        
    for i in values.keys():
        if values[i] == f_val:
            result.append(i)
        else:
            continue
    
    return result
    


	
goa=Series(np.random.normal(size=5))
goa[5]=100 # changing
del(goa[2]) # deleting

Z = np.random.random(10)
Z[Z.argmax()] = 0 # replace the maximum value by 0



# importance of index
	df = pd.DataFrame({'foo':np.random.random(10000),'key':range(100,10100)})
	%timeit df[df.key==10099] # following code performs the lookup repeatedly and reports on the performance
	df_with_index = df.set_index(['key'])
	%timeit df_with_index.loc[10099]


#---------------------------------------------------------------------------------------------------------------
# Analysing DF -------------------------------------------------------------------------------------------------

type(NQ100['lastsale']), SPY_TICK.dtypes
SPX500.shape, len(SPY_TICK)
GBPCAD.columns
SPX500.count(), SPY_TICK.describe()

# describe DF
	pip install pandas-profiling 
	import pandas_profiling
	df = pd.read_csv("titanic/train.csv")
	df.profile_report() # Show in NB
	profile = df.profile_report(title='Pandas Profiling Report')  
	profile.to_file(outputfile="Titanic data profiling.html")

	Z = np.random.random((5,5))
	Z = (Z - np.mean (Z)) / (np.std (Z)) # Normalize a 5x5 random matrix

# rename column	
	NQ100.rename(columns={'lastsale':'Last'}) 


# Deleting
	UC=USDCHF.dropna()

	# delete columns
		UC.drop(UC.columns[[3,4]],axis=1)
		interesting_collums = ['loyalty', 'satisfaction','educ']      
		reduced = data[interesting_collums]

		import itertools
		datecols = ['year', 'month', 'day']
		df = pd.DataFrame(list(itertools.product([2016,2017],[1,2],[1,2,3])),columns = datecols)
		df['data']=np.random.randn(len(df))
		df.index = pd.to_datetime(df[datecols])
		df=df.drop(datecols,axis=1).squeeze()

	# delete rows
		NQ100_small.drop('PYPL')
		NQ100new[-NQ100new.Last>1000]

# Appending rows
	UC_new= UC.nlargest(20,'Volume')
	UC_new.append(UC.nlargest(20,'Minute_ClCl'))
	pd.concat([UC.sample(n=10), UC.sample(n=10)])
	UC_new.loc['ZABR']=['Ukraine',100,120,1,2,3] # the number of columns should match

	d1 = {'a': 1}
	d2 = {'b': 2}
	d1.update(d2)
	print(d1)

	# Adding anomalies to df
	anomaly_dictionary={80: 3.1, 200: 3, 333: 1, 600: 2.6, 710: 2.1}
	gasoline_price_df.loc[:,'Artificially_Generated_Anomaly']=0
	for index, anomaly_value in anomaly_dictionary.items():
		gasoline_price_df.loc[index,'Gasoline_Price']=anomaly_value
		gasoline_price_df.loc[index,'Artificially_Generated_Anomaly']=1


	
# creating new columns
	NQ100['Capitalisation']=NQ100.Last*NQ100.share_volume
	NQ100['Random']=Series(np.random.normal(size=len(NQ100)),index=NQ100.index)
	NQ100.insert(1,'Rand',Series(np.random.normal(size=len(NQ100)),index=NQ100.index))
	NQ100.Randomize=NQ100.Rand
	USDCHF['Minute_ClCl']=USDCHF.Close.diff()

# Grouping
	values=np.random.randint(0,100,5)
	bins = pd.DataFrame({'Values':values})
	bins['Group']=pd.cut(values,range(0,101,10))
	
	def data_array_merge(data_array): # merge all dfs into one dfs    
		merged_df = functools.reduce(lambda left,right: pd.merge(left,right,on='Date'), data_array)
    	merged_df.set_index('Date', inplace=True)
    	return merged_df



        self.stocks = self.df['Symbol_Root'].unique()
        for stock in self.stocks:
            # Do aggregation
            stock_rows = self.df.loc[self.df['Symbol_Root'] == stock]




#making subsets
	USDCHF[USDCHF.Volume>200]
	NQ100[(NQ100.share_volume>10000000) & (NQ100.lastsale<40)]['Name']
	UC.nlargest(20,'Minute_ClCl')
	NQ100.nsmallest(4,'share_volume')['share_volume']
	NQ100.share_volume.nlargest(4)
	USDCHF[2:5]
	NQ100.sample(n=6)
	NQ100.loc['GOOG']
	temp_df['Hurra'][1:3]
	temp_df.iloc[1]
	market_data_250 = market_data.iloc[:250] # Select the first 250 rows
	NQ100.at['FB','Last']

	df[df["gender"] == "M"]["name"].nunique() # Unique names for male
	df[(df["M"] >= 50000) & (df["F"] >= 50000)] # names that atleast have 50,000 records for each gender

	male_df = df[df["gender"] == "M"].groupby("year").sum()
	male_df.min()["count"]
	male_df.idxmin()["count"]

	df[df["year"] >= 2008].pivot_table(index="name", 
										columns="year", 
										values="count", 
										aggfunc=np.sum).fillna(0)


	mask = df_results[pnl_col_name] > 0
	all_winning_trades = df_results[pnl_col_name].loc[mask] 
	
	
	# Select rows where Bid_Price>0 and Ask_Price>0 and Bid_Size>0 and Ask_Size>0
        training = training.ix[(training['Bid_Price']>0) | (training['Ask_Price']>0)]
        training = training.ix[(training['Bid_Size']>0) | (training['Ask_Size']>0)]

	# Only keep quotes at trading times
	df001 = df001.set_index('Date_Time')
	df001 = df001.between_time('9:30','16:00',include_start=True, include_end=True)


	''' Seperates dataframe into multiple by treatment
	E.g. if treatment is 'gender' with possible values 1 (male) or 2 (female) 
	the function returns a list of two frames: 
	1st - with all males, 2nd - with all females) '''
	def seperated_dataframes(df, treatment):
		treat_col = data[treatment] # col with the treatment
		dframes_sep = [] # list to hold seperated dataframes 
		for cat in categories(treat_col): # Go through all categories of the treatment
			df = data[treat_col == cat] # select all rows that match the category        
			dframes_sep.append(df) # append the selected dataframe
		return dframes_sep




	for ticker in stocks: # for each ticker in our pair          
		mask = (stock_data['Date'] > start_date) & (stock_data['Date'] <= end_date) # filter our column based on a date range   
		stock_data = stock_data.loc[mask] # rebuild our dataframe
		stock_data = stock_data.reset_index(drop=True) # re-index the data        
		array_pd_dfs.append(stock_data) # append our df to our array
	

	# Step by step approach, ...
		df = df[df["gender"] == "M"]
		df = df[["name", "count"]]
		df = df.groupby("name")
		df = df.sum()
		df = df.sort_values("count", ascending=False)
		df.head(10)
	# ... the same one-liner
	df[df["gender"] == "M"][["name", "count"]].groupby("name").sum().sort_values("count", ascending=False).head(10)
	

# Example of filtering the dataframe --------------------------------------------------------------------------------------
    def processData(self, date, major):
        impactful_data = self.data_df.loc[self.data_df['impact'] == 3].copy()
        impactful_data = self.data_df
        impactful_data['timestamp_af'] = impactful_data['timestamp'].apply(lambda x: self.utc_to_local(x))
        impactful_data['date'] = impactful_data['timestamp_af'].apply(lambda x: x[:10])
        impactful_data = impactful_data.loc[impactful_data['date'] == date].copy()
        impactful_data['major'] = impactful_data['economy'].apply(lambda x: True if x in major else False)
        impactful_data = impactful_data.loc[impactful_data['major'] == True].copy()
        self.final_data = impactful_data[['economy', 'name','impact','timestamp_af']].copy()	


# Example of filtering the dataframe --------------------------------------------------------------------------------------
	# Source: https://github.com/zbirnba1/quantative-finance/blob/master/src/recommended_portfolios.py
	qvdf=qvdf[pd.to_datetime(qvdf['release_date']).dt.date<last_valid_day.date()]
	qvdf=qvdf[pd.to_datetime(qvdf['end_date']).dt.date>=last_valid_day.date()-relativedelta(months=6)]
	qvdf=qvdf[qvdf['split_since_last_statement']==False]
	
	#filter out companies that will complicate my taxes
	qvdf=qvdf[~qvdf['name'].str[-2:].str.contains('LP')]
	qvdf=qvdf[~qvdf['name'].str[-3:].str.contains('LLC')]

	#Filter out Financial and Utilities
	s=qvdf['industry_category'].isin([None,"Banking","Financial Services","Real Estate","Utilities"])
	qvdf=qvdf[~s]

	#FILTER OUT MANIPULATORS OR DISTRESS COMPANIES: drop any companyes where either sta or snoa is nan
	qvdf = qvdf[((pd.notnull(qvdf['sta']))|(pd.notnull(qvdf['snoa'])))&
                	(pd.notnull(qvdf['pman']))&(pd.notnull(qvdf['pfd']))] #make sure one or the other is not nan
	qvdf = qvdf[(pd.notnull(qvdf['roa']))&(pd.notnull(qvdf['roc']))&
                	(pd.notnull(qvdf['cfoa']))&((pd.notnull(qvdf['mg']))|(pd.notnull(qvdf['ms'])))]

	if len(qvdf)==0:
		logging.error('empty qvdf')
		exit()
	qvdf=qvdf.sort_values(['sta'],na_position='last')#the lower the better
	totallen=len(qvdf[pd.notnull(qvdf['sta'])])
	i=1
	for index,row in qvdf[pd.notnull(qvdf['sta'])].iterrows():
		qvdf.loc[index,"p_sta"]=float(i)/float(totallen)
		i+=1

	qvdf=qvdf.sort_values(['snoa'],na_position='last') #the lower the better
	totallen=len(qvdf[pd.notnull(qvdf['snoa'])])
	i=1
	for index,row in qvdf[pd.notnull(qvdf['snoa'])].iterrows():
		qvdf.loc[index,"p_snoa"]=float(i)/float(totallen)
		i+=1

	qvdf['comboaccrual']=qvdf[["p_snoa","p_sta"]].mean(axis=1)
	qvdf=qvdf[pd.notnull(qvdf['comboaccrual'])]

	cutoff=.95
	s=(qvdf['comboaccrual']<cutoff)&(qvdf['p_pman']<cutoff)&(qvdf['p_pfd']<cutoff)
	qvdf=qvdf[s]

	qvdf = qvdf[(pd.notnull(qvdf['roa']))&(pd.notnull(qvdf['roc']))&(pd.notnull(qvdf['cfoa']))&
        	        ((pd.notnull(qvdf['mg']))|(pd.notnull(qvdf['ms'])))]

	qvdf['marginmax']=qvdf[["p_ms","p_mg"]].max(axis=1)
	qvdf['franchisepower']=qvdf[["marginmax","p_roa","p_roc","p_cfoa"]].mean(axis=1)
	qvdf=qvdf[pd.notnull(qvdf['franchisepower'])]

	qvdf=qvdf[pd.notnull(qvdf['emyield'])]
	qvdf=qvdf.sort_values(['emyield'],na_position='first') #the higher
	i=1
	for index,row in qvdf.iterrows():
		qvdf.loc[index,"p_emyield"]=float(i)/float(len(qvdf))
		i+=1
	s=qvdf['p_emyield']>=.9
	qvdf=qvdf[s]

	goodrows=(qvdf['newshares']<=0)|(qvdf['sec13']>0)|(qvdf['daystocover']<=1)|(qvdf['insider_purchase_ratio'].astype('float')>0)
	qvdf=qvdf[goodrows]
	qvdf['weight']=float(1)/float(len(qvdf))

	qvdf=qvdf[['ticker','name','industry_group','emyield','price','marketcap','weight']]
	qvdf=qvdf.set_index('ticker')


# Example of cleaning a dataframe ---------------------------------------------------------------------------

def clean_name(str_input): 
        if "<span" in str_input:
                soup = bs(str_input, "lxml")
                return soup.find('span')['onmouseover'].lstrip("tooltip.show('").rstrip(".');")
        return str_input

def clean_ticker(str_input):
        soup = bs(str_input, "lxml")
        return soup.find('a').text

def clean_allocation(str_input): 
        if str_input == "NA":
                return 0
        return float(str_input)/100

f['allocation'] = df.allocation.map(lambda x: clean_allocation(x))
df['name'] = df.name.map(lambda x: clean_name(x))
df['ticker'] = df.ticker.map(lambda x: clean_ticker(x))


import pandas as pd
import numpy as np


# Series creation
# Create dataframes 
# Import csv as df
# Selecting certain elements from series



# Series creation
		Series([1,2,3,4]), Series([3,4,5,6,7],index=['a','b','c','d','e']), Series([3]*5), np.zeros(10)
		Series(np.arange(4,9)) # using the numpy function
		Series(np.linspace(0,9,5)) # allows to specify the number of values to be created btw boundaries

		Series(np.random.normal(size=5))
		np.random.randint(50,101,len(dates))

		tm.makeFloatSeries(), tm.makeBoolIndex(), tm.makeCategoricalIndex()
		tm.makeCustomIndex(nentries=4,nlevels=2), tm.makeFloatIndex(), tm.makeIntIndex()
		tm.makeMultiIndex(), tm.makeRangeIndex(), tm.makeIntervalIndex()


# Create dataframes 
		dates=pd.date_range('2016-08-01','2017-08-01')
		s = Series(np.random.randint(50,60,size=len(dates)))
		temp_df=pd.DataFrame({'Hurra':dates, 'Pinguin':s})
		

		# create a dataframe
		PRICEDOMSIZE=  5  # domain size of prices
		SIZEDOMSIZE= 100
		def createTable(N):
			return pd.DataFrame({
					'pA': np.random.randint(0, PRICEDOMSIZE, N),
					'pB': np.random.randint(0, PRICEDOMSIZE, N),
					'sA': np.random.randint(0, SIZEDOMSIZE, N),
					'sB': np.random.randint(0, SIZEDOMSIZE, N)})


		# quickly create a dataframe for testing
		import pandas.util.testing as tm
		tm.N, tm.K = 5,3
		tm.makeDataFrame(), tm.makeMixedDataFrame(), tm.makeTimeDataFrame(freq="W")


# Importing data 
		SPX500=pd.read_csv("D:\\Data\\tick_data\\tick_data_zorro\\SPX500_2015.csv")
		SPY_TICK=pd.read_csv("D:\\Data\\tick_data\\SPY_TICK_TRADE.csv")
		GBPCAD=pd.read_csv("D:\\Data\\minute_data\\GBPCAD_2017.csv", header=False, parse_dates=['Date'])
		NQ100=pd.read_csv("http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx?render=download",
						usecols=[0,1,2,5],
						index_col='Symbol',
						skipinitialspace=True)

		
		df=pd.DataFrame(pd.read_csv(file_path))


		import pandas as pd
		pd.read_csv('http://www.nasdaq.com/investing/etfs/etf-finder-results.aspx?download=Yes')['Symbol'].values


		calls_df, = pd.read_html("http://apps.sandiego.gov/sdfiredispatch/", header=0, parse_dates=["Call Date"])


		spyderdat = pd.read_csv("/home/curtis/Downloads/HistoricalQuotes.csv")
		spyderdat = pd.DataFrame(spyderdat.loc[:, ["open", "high", "low", "close", "close"]]
								.iloc[1:].as_matrix(),
								index=pd.DatetimeIndex(spyderdat.iloc[1:, 0]),
								columns=["Open", "High", "Low", "Close", "Adj Close"])
					.sort_index()
		spyder = spyderdat.loc[start:end]
		stocks = stocks.join(spyder.loc[:, "Adj Close"])
						.rename(columns={"Adj Close": "SPY"})




# Selecting certain elements from series --------------------------------------

		s = Series(np.random.randint(10,100,size =30))
		s[3], s[[1,3]], s[3:17:4] # step 4
		s.head(), s[:5], s[:-2] # all but the last 2


		bob = Series(np.arange(3,30,3))
		bob >15
		bob[(bob>15) & (bob<25)]


		line = Series(np.random.randint(1,200,size=1000))
		line.sample(n=3)
		line.sample(frac=0.05) #selects 5% of data


		lst = [40, 10, 20, 30]
		min(range(len(lst)), key=lst.__getitem__) # Find Index of Min Element

		a = np.array([2,4,6,9,4])
		np.argmax(a)
		np.argwhere(a==4)


		# if you suggest 20%, it will neglect the best 10% of values
		# and the worst 10% of values.
			def trimmean(arr, per):
				ratio = per/200 # /100 for easy calculation by *, and /2 for easy adaption to best and worst parts.
				cal_sum = 0 # sum value to be calculated to trimmean.
				arr.sort()
				neg_val = int(len(arr)*ratio)
				arr = arr[neg_val:len(arr)-neg_val]
				for i in arr:
					cal_sum += i
				return cal_sum/len(arr)


		# If array, Min, Max value was given, it returns array that contains
		# values of given array which was larger than Min, and lower than Max.
		def limit(arr, min_lim=None, max_lim=None):
			min_check = lambda val: True if min_lim is None else (min_lim <= val)
			max_check = lambda val: True if max_lim is None else (val <= max_lim)    
			return [val for val in arr if min_check(val) and max_check(val)]






# Analysing series -------------------------------------------------------

		bob = Series(np.arange(3,30,3))
		(bob>10).any() # bool
		(bob>2).all() # bool
		(bob>15).sum()
		bob.isnull().sum()

		np.nonzero([1,2,0,0,4,0]) # find indices of non-zero elements


		s = input()
		up    = len([i for i in ['1', '2', '3'] if i in s]) ==  0 # true if s does not contains one of these
		right = len([i for i in ['3', '6', '9', '0'] if i in s]) == 0
		left  = len([i for i in ['1', '4', '7', '0'] if i in s]) == 0
		down  = len([i for i in ['7', '9', '0'] if i in s]) == 0
		ok = not(up or right or left or down) #if the is at least one true then it is not uniqe and print NO
												#if all of them are 0 then not 0 == true and print YES
		if ok:
			print("YES")
		else: #not ok
			print("NO")


		# the most often occuring names using collection.Counter
		from collections import Counter
		cheese = ["gouda", "brie", "feta", "cream cheese", "feta", "cheddar",
				"parmesan", "parmesan", "cheddar", "mozzarella", "cheddar", "gouda",
				"parmesan", "camembert", "emmental", "camembert", "parmesan"]
		cheese_count = Counter(cheese) # maps items to number of occurrences
		# use update(more_words) method to easily add more elements to counter
		print(cheese_count.most_common(3))
		# Prints: [('parmesan', 4), ('cheddar', 3), ('gouda', 2)]



		# Takes in an array of numbers and finds consecutive runs of the number to_find
		def find_runs(arr, to_find):
			# Create an array that is 1 where arr is equal to to_find, and pad each end with an extra 0.
			is_the_number = np.concatenate(([0], np.equal(arr, to_find).view(np.int8), [0]))
			absdiff = np.abs(np.diff(is_the_number))
			ranges = np.where(absdiff == 1)[0].reshape(-1, 2) # Runs start and end where absdiff is 1.
			return ranges

			if __name__ == "__main__":
				test_arr = [1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 9, 8, 7, 0, 10, 11]
				print("find_runs() output: ")
				print(find_runs(test_arr, 0))
				print("Array outputted should be equal to: ")
				print("[[3, 9], [12, 16], [19, 20]]")





# Comparing several series -------------------------------------------------------

		# find the closest value (to a given scalar)
		Z = np.arange(100)
		v = np.random.uniform(0,100)
		index = (np.abs(Z-v)).argmin()
		print(Z[index])

		# find common values between two arrays
		Z1 = np.random.randint(0,10,10)
		Z2 = np.random.randint(0,10,10)
		print(np.intersect1d(Z1,Z2))

		# find common values between two arrays
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



# Changing series -------------------------------------------------------

		goa=Series(np.random.normal(size=5))
		goa[5]=100 # changing
		del(goa[2]) # deleting

		Z = np.random.random(10)
		Z[Z.argmax()] = 0 # replace the maximum value by 0


		d1 = {'a': 1}
		d2 = {'b': 2}
		d1.update(d2)
		print(d1)



# describe DF ---------------------------------------------------------------------

		type(NQ100['lastsale']), SPY_TICK.dtypes
		SPX500.shape, len(SPY_TICK)
		GBPCAD.columns
		SPX500.count(), SPY_TICK.describe()

		df.groupby("continent")["beer_servings"].describe()


		# describe DF
		pip install pandas-profiling 
		import pandas_profiling
		df = pd.read_csv("titanic/train.csv")
		df.profile_report() # Show in NB
		profile = df.profile_report(title='Pandas Profiling Report')  
		profile.to_file(outputfile="Titanic data profiling.html")


		# memory usage
			df = generate_sample_data_datetime().reset_index()
			df.columns = ["date", "sales", "customers"]
			df.info(memory_usage = "deep") # Show the global usage of memory of the df"
			df.memory_usage(deep = True) # Show the usage of memory of every column




# Iterations thorugh the rows -------------------------------------------------------

		# never use iterrows
		# rarely use intertuples:
				result = 0
				for(_, col1, col2, col3, col4) in df.itertuples(name=None): 
					result += max(col2, col3)


		# list comprehensions are also useful:
				result = [f(x) for x in df['col']] # iterating over one column - `f` is some function that processes your data
				result = [f(x, y) for x, y in zip(df['col1'], df['col2'])] # iterating over two columns, use `zip`

				squares = [i * i for i in range(10)] 

				txns = [1.09, 23.56, 57.84, 4.56, 6.78]
				TAX_RATE = .08
				def get_price_with_tax(txn):
					return txn * (1 + TAX_RATE)
				final_prices = [get_price_with_tax(i) for i in txns]


				sentence = 'the rocket came back from mars'
				vowels = [i for i in sentence if i in 'aeiou'] # list
				unique_vowels = {i for i in sentence if i in 'aeiou'} # set


				original_prices = [1.25, -9.45, 10.22, 3.78, -5.92, 1.16]
				prices = [i if i > 0 else 0 for i in original_prices]


		# map reduce
				from more_itertools import map_reduce
				data = 'This sentence has words of various lengths in it, both short ones and long ones'.split()

				keyfunc = lambda x: len(x)
				result = map_reduce(data, keyfunc)
				# defaultdict(None, {
				#   4: ['This', 'both', 'ones', 'long', 'ones'],
				#   8: ['sentence'],
				#   3: ['has', 'it,', 'and'],
				#   5: ['words', 'short'],
				#   2: ['of', 'in'],
				#   7: ['various', 'lengths']})






# changes to DF ---------------------------------------------------------------------


		NQ100.rename(columns={'lastsale':'Last'}) # rename column	

		# add a prefix or suffix to all columns
		df.add_prefix("1_")
		df.add_suffix("_Z")

		data['educ'] = pd.to_numeric(data['educ'],errors='coerce') # errors='coerce' means that we force the conversation. noncovertable are set to NaN


		# Deleting

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
				df.dropna(axis = "columns") # drop any column that has missing values
				df.dropna(thresh = len(df)*0.95, axis = "columns") # drop column where missing values are above a threshold


			# delete rows
				NQ100_small.drop('PYPL')
				NQ100new[-NQ100new.Last>1000]
				UC=USDCHF.dropna()
				df.isna().mean() # calculate the % of missing values in each row
				df1.dropna(axis = "rows") # drop any row that has missing values





		# Appending rows
			UC_new= UC.nlargest(20,'Volume')
			UC_new.append(UC.nlargest(20,'Minute_ClCl'))

			UC_new.loc['ZABR']=['Ukraine',100,120,1,2,3] # the number of columns should match

			ho = ['ON']  # overnight
			ho.extend([str(i) + 'W' for i in range(1,4)])  # weekly tenors
			ho.extend([str(i) + 'M' for i in range(1,12)])  # monthly tenors
			ho.extend([str(i) + 'Y' for i in range(1,51)])  # yearly tenors

			# Adding anomalies to df
			anomaly_dictionary={80: 3.1, 200: 3, 333: 1, 600: 2.6, 710: 2.1}
			gasoline_price_df.loc[:,'Artificially_Generated_Anomaly']=0
			for index, anomaly_value in anomaly_dictionary.items():
				gasoline_price_df.loc[index,'Gasoline_Price']=anomaly_value
				gasoline_price_df.loc[index,'Artificially_Generated_Anomaly']=1


			# see where the columns are coming from
			pd.merge(df, df1, how = "left", indicator = True)



		# creating new columns
			NQ100['Capitalisation']=NQ100.Last*NQ100.share_volume
			NQ100['Random']=Series(np.random.normal(size=len(NQ100)),index=NQ100.index)
			NQ100.insert(1,'Rand',Series(np.random.normal(size=len(NQ100)),index=NQ100.index))
			NQ100.Randomize=NQ100.Rand
			USDCHF['Minute_ClCl']=USDCHF.Close.diff()
			df['CohortIndex_d'] = (df['last_active_date'] - df['signup_date']).dt.days # new column with the difference between the two dates
			gasoline_price_df.loc[:,'Artificially_Generated_Anomaly']=0

			# split column into 2 columns
			df[[one,two]] = df[orig].str.split(separator,expand=True)

			# select columns:
			list(my_dataframe)
			my_dataframe.columns.values.tolist()


			# split column with list
			d = {"A":[1, 2, 3], "B":[[10, 20], [40, 50], [60, 70]]}
			df = pd.DataFrame(d)
			df_ = df["B"].apply(pd.Series) # Convert it to normal series
			pd.merge(df, df_, left_index = True, right_index = True)


		# Remove a column and store it as a separate series
			meta = df.pop("Metascore").to_frame() 


# Extracting sub-set from DF --------------------------------------------------------------

		UC_new= UC.nlargest(20,'Volume')
		NQ100.nsmallest(4,'share_volume')['share_volume']
		NQ100.share_volume.nlargest(4)

		USDCHF[USDCHF.Volume>200]
		NQ100[(NQ100.share_volume>10000000) & (NQ100.lastsale<40)]['Name']
		USDCHF[2:5]
		NQ100.sample(n=6)
		NQ100.loc['GOOG']
		temp_df['Hurra'][1:3]
		temp_df.iloc[1]
		market_data_250 = market_data.iloc[:250] # Select the first 250 rows
		NQ100.at['FB','Last']


		df[df["gender"] == "M"]["name"].nunique() # Unique names for male
		df[(df["M"] >= 50000) & (df["F"] >= 50000)] # names that atleast have 50,000 records for each gender

		# select columns with f-string
		df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv")
		drink = "wine"
		df[f'{drink}_servings'].to_frame() # allows us to iterate fast over columns


		# Select columns by dtype
			df.select_dtypes(include = "number") # Select numerical columns
			df.select_dtypes(include = "object") # Select string columns
			df.select_dtypes(include = ["datetime", "timedelta"]) # Select datetime columns
			df.select_dtypes(include = ["int8", "int16", "int32", "int64", "float"]) # Select by passing the dtypes you need




		# Different methods:
			df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
							'B': 'one one two three two two one three'.split(),
							'C': np.arange(8), 'D': np.arange(8) * 2})
			df.loc[df['A'] == 'foo']

			df.loc[df['B'].isin(['one','three'])]

			df = df.set_index(['B']) # if you wish to do this many times, it is more efficient to make an index first
			df.loc['one']

			mask = df['A'] == 'foo'
			df[mask]


		# Filter only the largest categories
			df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")
			df.columns = map(str.lower, list(df.columns)) # convert headers to lower type
			top_genre = df["genre"].value_counts().to_frame()[0:3].index # select top 3 genre
			df_top = df[df["genre"].isin(top_genre)] # now let's filter the df with the top genre
			



		male_df = df[df["gender"] == "M"].groupby("year").sum()
		male_df.min()["count"]
		male_df.idxmin()["count"]

		df[df["year"] >= 2008].pivot_table(index="name", 
											columns="year", 
											values="count", 
											aggfunc=np.sum).fillna(0)


		mask = df_results[pnl_col_name] > 0
		all_winning_trades = df_results[pnl_col_name].loc[mask] 


		def clean_allocation(str_input): 
				if str_input == "NA":
						return 0
				return float(str_input)/100
		df['allocation'] = df.allocation.map(lambda x: clean_allocation(x))


		# Select rows where Bid_Price>0 and Ask_Price>0 and Bid_Size>0 and Ask_Size>0
			training = training.ix[(training['Bid_Price']>0) | (training['Ask_Price']>0)]
			training = training.ix[(training['Bid_Size']>0) | (training['Ask_Size']>0)]

		# Only keep quotes at trading times
		df001 = df001.set_index('Date_Time')
		df001 = df001.between_time('9:30','16:00',include_start=True, include_end=True)


		# Split a df into 2 random subsets
		df_1 = df.sample(frac = 0.7)
		df_2 = df.drop(df_1.index) # only works if the df index is unique




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


		pd.concat([UC.sample(n=10), UC.sample(n=10)])

		# importance of index
		df = pd.DataFrame({'foo':np.random.random(10000),'key':range(100,10100)})
		%timeit df[df.key==10099] # following code performs the lookup repeatedly and reports on the performance
		df_with_index = df.set_index(['key'])
		%timeit df_with_index.loc[10099]



# Grouping
		values=np.random.randint(0,100,5)
		bins = pd.DataFrame({'Values':values})
		bins['Group']=pd.cut(values,range(0,101,10))

		male_df = df[df["gender"] == "M"].groupby("year").sum()
		
		df.resample("M")["sales"].sum() # groupby by month


		# Combine the output of an aggregation with the original df
			d = {"orderid":[1, 1, 1, 2, 2, 3, 4, 5], 
				 "item":[10, 120, 130, 200, 300, 550, 12.3, 200],
				 "salesperson":["Nico", "Carlos", "Juan", "Nico", "Nico", "Juan", "Maria", "Carlos"]}
			df = pd.DataFrame(d)
			df["total_items_sold"] = df.groupby("orderid")["item"].transform(sum) 
			df["running_total"] = df["item"].cumsum()
			df["running_total_by_person"] = df.groupby("salesperson")["item"].cumsum()


		def data_array_merge(data_array): # merge all dfs into one dfs    
			merged_df = functools.reduce(lambda left,right: pd.merge(left,right,on='Date'), data_array)
			merged_df.set_index('Date', inplace=True)
			return merged_df



			self.stocks = self.df['Symbol_Root'].unique()
			for stock in self.stocks:
				# Do aggregation
				stock_rows = self.df.loc[self.df['Symbol_Root'] == stock]



		# Selecting unique values from dataframe: the quickest is via numpy

			df = pd.DataFrame({'Col1': ['Bob', 'Joe', 'Bill', 'Mary', 'Joe'],
							'Col2': ['Joe', 'Steve', 'Bob', 'Bob', 'Steve'],
							'Col3': np.random.random(5)})
			np.unique(df[['Col1', 'Col2']].values) # array(['Bill', 'Bob', 'Joe', 'Mary', 'Steve'], dtype=object)
			set(np.concatenate(df.values))



		# Combine the small categories into a single category named "Others"
			d = {"genre": ["A", "A", "A", "A", "A", "B", "B", "C", "D", "E", "F"]}
			df = pd.DataFrame(d)

			# 1st way
			frequencies = df["genre"].value_counts(normalize = True) # Step 1: count the frequencies
			threshold = 0.1
			small_categories = frequencies[frequencies < threshold].index # Step 2: filter the smaller categories
			df["genre"] = df["genre"].replace(small_categories, "Other") # Step 3: replace the values
			df["genre"].value_counts(normalize = True)

			# 2nd way
			top_four = df["genre"].value_counts().nlargest(4).index
			df_updated = df.where(df["genre"].isin(top_four), other = "Other")
			df_updated["genre"].value_counts()


			# Convert continuos variable to categorical (cut and qcut)
				df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")
				pd.cut(df["Metascore"], bins = [0, 25, 50, 75, 99]).head() # Using cut you can specify the bin edges
				pd.qcut(df["Metascore"], q = 3).head() # specify the number of bins
				pd.qcut(df["Metascore"], q = 4, labels = ["awful", "bad", "average", "good"]).head() # cut and qcut accept label bin size


	

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




# Pairwise iteration btw columns of dataframe ------------------------------------------------------------------------------

from scipy.stats import ttest_ind
from itertools import combinations

N, M = 20, 4
A = np.random.randn(N, M) + np.arange(M)/4 # generate a random array, add a small constant to each column
df = pd.DataFrame(A) # converts numpy array to pandas df
pairwise_pvalues = pd.DataFrame(columns=df.columns, index=df.columns, dtype=float)
for (label1, column1), (label2, column2) in combinations(df.items(), 2):
    pairwise_pvalues.loc[label1, label2] = ttest_ind(column1, column2)[1]
    pairwise_pvalues.loc[label2, label1] = pairwise_pvalues.loc[label1, label2]
pairwise_pvalues.round(3)


# --------------------------------------------------------------------------------
# Permutations

from itertools import permutations 
my_list = [1,2,3]
perm = list(permutations(my_list))

#(1, 2, 3)
#(1, 3, 2)
#(2, 1, 3)
#(2, 3, 1)
#(3, 1, 2)
#(3, 2, 1)

#--------------------------------------------------------------------------------

# Detect and Replace Missing Values
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

# -------------------------------------------------------------------------------------

color_list = ["green", "red", "blue", "yellow"]
rgb = [color for color in color_list if color in('green', 'red', 'blue')]
rgb



green_list = [color for color in color_list if color == 'green']

green_list2 = []
for color in color_list:
    if color == 'green':
        green_list2.append(color)


color_indicator = [0 if color == 'green'else 1 if color == 'red' else 2 if color == 'blue' else 3 for color in color_list]
print(color_list)
print(color_indicator)

color_mapping = {'green': 0, 'red': 1, 'blue':2, 'yellow':3}
color_indicator2 = [color_mapping[color] if color in color_mapping else 'na' for color in color_list]
print(color_list)
print(color_indicator2)

word_lengths = [len(color) for color in color_list]
word_lengths

color_list1 = ['green', 'red', 'blue', 'yellow']
color_list2 = ['dark', 'bright', 'tinted', 'glowing']

color_matrix = [[color2 + ' ' + color1 for color1 in color_list1] for color2 in color_list2]
color_matrix


# ---------------------------------------------------------------------------------------

# Bid prices offered by the two buyers, pA and pB. Bid sizes, sA and sB. 
# Add a new best size column (bS) to the table, that returns the size at the best price. 
# If the two buyers have the same price then bS is equal to sA + sB


# Let's assume we have only 2 buyers:

PRICEDOMSIZE=  5  # domain size of prices
SIZEDOMSIZE= 100
N = 1000 * 1000

def createTable(N):
	return pd.DataFrame({
			'pA': np.random.randint(0, PRICEDOMSIZE, N),
			'pB': np.random.randint(0, PRICEDOMSIZE, N),
			'sA': np.random.randint(0, SIZEDOMSIZE, N),
			'sB': np.random.randint(0, SIZEDOMSIZE, N)})

t = createTable(N)


# 1st approach: go row by row
def bestSizeIF(pA, pB, sA, sB):
    if pA == pB:
        return sA + sB
    return sA if pA > pB else sB

t['bS']= t.apply(lambda row: bestSizeIF(row['pA'], row['pB'], row['sA'], row['sB']), axis=1)

# vectorize approach with True/False vector
def bestSizeWHERE(pA, pB, sA, sB):
    p = np.array([pA, pB])
    return np.sum(np.array([sA, sB])[
				np.where(p == np.max(p))]) # where returns list of indices containing True values

t['bS']= t.apply(lambda row: bestSizeWHERE(row['pA'], row['pB'], row['sA'], row['sB']), axis=1)

def bestSizeMULT(pA, pB, sA, sB):
    p = np.array([pA, pB])
    return np.sum(np.array([sA, sB]) *
        (p == np.max(p))) # Boolean arithmetic in which True acts as one, False acts as zero

	def bestSizeMULT2(pA, pB, sA, sB):
		return sA * (pA >= pB) + sB * (pA <= pB) # another approach

t['bS']= t.apply(lambda row: bestSizeMULT(row['pA'], row['pB'], row['sA'], row['sB']), axis=1)




# --------------------------------------------------------------------------------


from more_itertools import partition
# Split based on file extension
files = [
    "foo.jpg",
    "bar.exe",
    "baz.gif",
    "text.txt",
    "data.bin",
]

ALLOWED_EXTENSIONS = ('jpg','jpeg','gif','bmp','png')
is_allowed = lambda x: x.split(".")[1] in ALLOWED_EXTENSIONS

allowed, forbidden = partition(is_allowed, files)
list(allowed)
#  ['bar.exe', 'text.txt', 'data.bin']
list(forbidden)
#  ['foo.jpg', 'baz.gif']


# ------------------------------------------------------------------------------------



#########################################################


def all_equal(lst):
      return lst[1:] == lst[:-1]

all_equal([1, 2, 3, 4, 5, 6]) # False
all_equal([1, 1, 1, 1]) # True


def all_unique(lst):
      return len(lst) == len(set(lst))

x = [1, 2, 3, 4, 5, 6]
y = [1, 2, 2, 3, 4, 5]
all_unique(x) # True
all_unique(y) # False


def difference(a, b):
      _b = set(b)
  return [item for item in a if item not in _b]

difference([1, 2, 3], [1, 2, 4]) # [3]



def count_occurrences(lst, val):
      return len([x for x in lst if x == val and type(x) == type(val)])

count_occurrences([1, 1, 2, 1, 2, 3], 1) # 3


from time import sleep

def delay(fn, ms, *args):
  sleep(ms / 1000)
  return fn(*args)

delay(lambda x: print(x),1000,'later') # prints 'later' after one second


def digitize(n):
      return list(map(int, str(n)))

digitize(123) # [1, 2, 3]


from collections import Counter
def filter_non_unique(lst):
  return [item for item, count in counter = Counter(lst).items() if count == 1]

filter_non_unique([1, 2, 2, 3, 4, 4, 5]) # [1, 3, 5]




##########################################################################################################

# Allign dataseries

def datetime_aligned(ds1, ds2, maxLen=None):
    """
    Returns two dataseries that exhibit only those values whose datetimes are in both dataseries.

    :param ds1: A DataSeries instance.
    :type ds1: :class:`DataSeries`.
    :param ds2: A DataSeries instance.
    :type ds2: :class:`DataSeries`.
    :param maxLen: The maximum number of values to hold for the returned :class:`DataSeries`.
        Once a bounded length is full, when new items are added, a corresponding number of items are discarded from the
        opposite end. If None then dataseries.DEFAULT_MAX_LEN is used.
    :type maxLen: int.
    """
    aligned1 = dataseries.SequenceDataSeries(maxLen)
    aligned2 = dataseries.SequenceDataSeries(maxLen)
    Syncer(ds1, ds2, aligned1, aligned2)
    return (aligned1, aligned2)


# This class is responsible for filling 2 dataseries when 2 other dataseries get new values.
class Syncer(object):
    def __init__(self, sourceDS1, sourceDS2, destDS1, destDS2):
        self.__values1 = []  # (datetime, value)
        self.__values2 = []  # (datetime, value)
        self.__destDS1 = destDS1
        self.__destDS2 = destDS2
        sourceDS1.getNewValueEvent().subscribe(self.__onNewValue1)
        sourceDS2.getNewValueEvent().subscribe(self.__onNewValue2)
        # Source dataseries will keep a reference to self and that will prevent from getting this destroyed.

    # Scan backwards for the position of dateTime in ds.
    def __findPosForDateTime(self, values, dateTime):
        ret = None
        i = len(values) - 1
        while i >= 0:
            if values[i][0] == dateTime:
                ret = i
                break
            elif values[i][0] < dateTime:
                break
            i -= 1
        return ret

    def __onNewValue1(self, dataSeries, dateTime, value):
        pos2 = self.__findPosForDateTime(self.__values2, dateTime)
        # If a value for dateTime was added to first dataseries, and a value for that same datetime is also in the second one
        # then append to both destination dataseries.
        if pos2 is not None:
            self.__append(dateTime, value, self.__values2[pos2][1])
            # Reset buffers.
            self.__values1 = []
            self.__values2 = self.__values2[pos2+1:]
        else:
            # Since source dataseries may not hold all the values we need, we need to buffer manually.
            self.__values1.append((dateTime, value))

    def __onNewValue2(self, dataSeries, dateTime, value):
        pos1 = self.__findPosForDateTime(self.__values1, dateTime)
        # If a value for dateTime was added to second dataseries, and a value for that same datetime is also in the first one
        # then append to both destination dataseries.
        if pos1 is not None:
            self.__append(dateTime, self.__values1[pos1][1], value)
            # Reset buffers.
            self.__values1 = self.__values1[pos1+1:]
            self.__values2 = []
        else:
            # Since source dataseries may not hold all the values we need, we need to buffer manually.
            self.__values2.append((dateTime, value))

    def __append(self, dateTime, value1, value2):
        self.__destDS1.appendWithDateTime(dateTime, value1)
        self.__destDS2.appendWithDateTime(dateTime, value2)

# Series creation
Series([1,2,3,4])
Series([3,4,5,6,7],index=['a','b','c','d','e'])
Series([3]*5)
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
tm.makeTimedeltaIndex()
tm.makeTimeSeries()
tm.makePeriodSeries()
tm.makeDateIndex()
tm.makePeriodIndex()
tm.makeObjectSeries()

# random series
Series(np.random.normal(size=5))
np.random.randint(50,101,len(dates))

import pandas.util.testing as tm
tm.N, tm.K = 5,3
tm.makeFloatSeries()
tm.makeStringSeries()
tm.makeBoolIndex()
tm.makeCategoricalIndex()
tm.makeCustomIndex(nentries=4,nlevels=2)
tm.makeFloatIndex()
tm.makeIntIndex()
tm.makeIntervalIndex()
tm.makeMultiIndex()
tm.makeRangeIndex()

# string series
Series(list('abcde'))
random.choices(string.ascii_lowercase,k=5) # generates k random letters
tm.makeStringIndex()

# Selection of a series member -----------------------------------------------------------------------------

s = Series(np.random.randint(10,100,size =30))
s[3], s[[1,3]], s[3:17:4] # step 4
s.head(), s[:5], s[:-2] # all but the last 2

bob = Series(np.arange(3,30,3))
bob >15
bob[(bob>15) & (bob<25)]

line = Series(np.random.randint(1,200,size=1000))
line.sample(n=3)
line.sample(frac=0.05) #selects 5% of data


# Actions with series -------------------------------------------------------------------------------------


s= Series(np.random.randint(50,60,size=20))
s.values, s.index
len(s), s.size
s.unique(), s.value_counts(), s.nunique()# number of unique values
s.mean(), s.describe(), s.idxmax()
s.rank()

bob = Series(np.arange(3,30,3))
(bob>10).any() # bool
(bob>2).all() # bool
(bob>15).sum()
bob.isnull().sum()


# rolling window
s = Series(np.random.randint(1,200,size=1000))
r=s.rolling(window=10)
s.plot()
r.mean().plot()

goa=Series(np.random.normal(size=5))
goa[5]=100 # changing
del(goa[2]) # deleting

# Dataframes (many series) -----------------------------------------------------------------------

dates=pd.date_range('2016-08-01','2017-08-01')
s = Series(np.random.randint(50,60,size=len(dates)))
temp_df=pd.DataFrame({'Hurra':dates, 'Pinguin':s})

from itertools import product
datecols = ['year', 'month', 'day']
df = pd.DataFrame(list(product([2016,2017],[1,2],[1,2,3])),columns = datecols)
df['data']=np.random.randn(len(df))
df.index = pd.to_datetime(df[datecols])

# quickly create a dataframe for testing
import pandas.util.testing as tm
tm.N, tm.K = 5,3
tm.makeDataFrame()
tm.makeMixedDataFrame()
tm.makeTimeDataFrame(freq="W")

# Importing data -----------------------------------------------------------------------
SPX500=pd.read_csv("D:\\Data\\tick_data\\tick_data_zorro\\SPX500_2015.csv")
SPY_TICK=pd.read_csv("D:\\Data\\tick_data\\SPY_TICK_TRADE.csv")
GBPCAD=pd.read_csv("D:\\Data\\minute_data\\GBPCAD_2017.csv", header=False, parse_dates=['Date'])
NQ100=pd.read_csv("http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx?render=download",
                 usecols=[0,1,2,5],
                 index_col='Symbol',
                 skipinitialspace=True)


calls_df, = pd.read_html("http://apps.sandiego.gov/sdfiredispatch/", header=0, parse_dates=["Call Date"])

#errors='coerce' means that we force the conversation.
#Values that can not be converted are set to NaN ("Not a Number")
data['educ'] = pd.to_numeric(data['educ'],errors='coerce') 




# Analysing DF --------------------------------------------------------------------------------
type(NQ100['lastsale']), SPY_TICK.dtypes
SPX500.shape, len(SPY_TICK)
GBPCAD.columns
SPX500.count(), SPY_TICK.describe()

NQ100.rename(columns={'lastsale':'Last'}) # rename column

# creating new columns
NQ100['Capitalisation']=NQ100.Last*NQ100.share_volume
NQ100['Random']=Series(np.random.normal(size=len(NQ100)),index=NQ100.index)
NQ100.insert(1,'Rand',Series(np.random.normal(size=len(NQ100)),index=NQ100.index))
NQ100.Randomize=NQ100.Rand
USDCHF['Minute_ClCl']=USDCHF.Close.diff()

UC=USDCHF.dropna()


# Grouping
values=np.random.randint(0,100,5)
bins = pd.DataFrame({'Values':values})
bins['Group']=pd.cut(values,range(0,101,10))

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
NQ100.at['FB','Last']

df[df["gender"] == "M"]["name"].nunique() # Unique names for male
df[(df["M"] >= 50000) & (df["F"] >= 50000)] # names that atleast have 50,000 records for each gender

male_df = df[df["gender"] == "M"].groupby("year").sum()
male_df.min()["count"]
male_df.idxmin()["count"]

df[df["year"] >= 2008].pivot_table(index="name", columns="year", values="count", aggfunc=np.sum).fillna(0)



# Step by step approach, ...
df = df[df["gender"] == "M"]
df = df[["name", "count"]]
df = df.groupby("name")
df = df.sum()
df = df.sort_values("count", ascending=False)
df.head(10)
# ... the same one-liner
df[df["gender"] == "M"][["name", "count"]].groupby("name").sum().sort_values("count", ascending=False).head(10)




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


# importance of index
df = pd.DataFrame({'foo':np.random.random(10000),'key':range(100,10100)})
%timeit df[df.key==10099] # following code performs the lookup repeatedly and reports on the performance
df_with_index = df.set_index(['key'])
%timeit df_with_index.loc[10099]

# Working with dates -----------------------------------------------------------------------------------

pd.to_datetime(pd.Series(["Jul 31, 2017","2010-10-01","2016/10/10","2014.06.10"]))
pd.to_datetime(pd.Series(["11 Jul 2018","13.04.2015","30/12/2011"]),dayfirst=True)
# providing a format could increase speed of conversion significantly
pd.to_datetime(pd.Series(["12-11-2010 01:56","11-01-2012 22:10","28-02-2013 14:59"]), format='%d-%m-%Y %H:%M')

# epoch timestamps: default unit is nanoseconds
pd.to_datetime([1349720105100, 1349720105200, 1349720105300, 1349720105400, 1349720105500], unit='ms')

start=pd.Timestamp("2018-01-06 00:00:00")
pd.date_range(start, periods=10,freq="2h20min")

# Business days
start = datetime(2018,10,1), end = datetime(2018,10,10)
pd.date_range(start,end)
pd.bdate_range(start,end)
pd.bdate_range(start,periods=4,freq="BQS")

rng=pd.date_range(start,end,freq="BM")
ts=pd.Series(np.random.randn(len(rng)),index=rng)
ts["2018"]
ts["2019-2":"2019-7"]
ts.truncate(before="2019-2",after="2019-7") # select less than above

# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components

two_biz_days=2*pd.offsets.BDay()
friday = pd.Timestamp("2018-01-05")
friday.day_name()
two_biz_days.apply(friday).day_name()
(friday+two_biz_days),(friday+two_biz_days).day_name()

ts =pd.Timestamp("2018-01-06 00:00:00")
ts.day_name() # --> "Saturday"
offset=pd.offsets.BusinessHour(start="09:00")
offset.rollforward(ts) # Bring the date to the closest offset date (Monday)

pd.offsets.BusinessHour() # from 9 till 17
rng = pd.date_range("2018-01-10","2018-01-15",freq="BH") # BH is "business hour"
rng+pd.DateOffset(months=2,hours=3)

rng=pd.date_range(start,end,freq="D")
ts=pd.Series(np.random.randn(len(rng)),index=rng)
ts.shift(2)[1]

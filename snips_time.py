


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


	from itertools import product
	datecols = ['year', 'month', 'day']
	df = pd.DataFrame(list(product([2016,2017],[1,2],[1,2,3])),columns = datecols)
	df['data']=np.random.randn(len(df))
	df.index = pd.to_datetime(df[datecols])


# -------------------------------------------------------------------------------------------------------------

day = re.compile('[1-3][0-9]?') # day regex
year = re.compile('20[0-9]{2}') # year regex
    
tmp = self.soup.find('small', text=re.compile('market', re.IGNORECASE)).text.split('Market')[0].strip()
self.year = year.search(tmp).group(0)
self.day = day.search(tmp).group(0)
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
for ii, mo in enumerate(months, 1): # iterate over months and flag if match found
    more = re.compile(mo, re.IGNORECASE)
    if more.search(tmp):
        self.month = ii
        break
	

# -------------------------------------------------------------------------------------------------------------

# Only keep quotes at trading times
df001 = df001.set_index('Date_Time')
df001 = df001.between_time('9:30','16:00',include_start=True, include_end=True)

# -------------------------------------------------------------------------------------------------------------
# 

# biz days btw 2 dates

            # http://dateutil.readthedocs.io/en/stable/rrule.html
            from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR
            def daterange(start_date, end_date):
                # automate a range of business days between two dates
                return rrule(DAILY, dtstart=start_date, until=end_date, byweekday=(MO,TU,WE,TH,FR))

            for tr_date in daterange(start_date, end_date):


# itertools through biz days

            now   = dt.date.today()
            year  = str(now.year)
            m     = str(now.month)
            month = '0'+m

            day_5 = now - 5 * BDay()
            day_4 = now - 4 * BDay()
            day_3 = now - 3 * BDay()
            day_2 = now - 2 * BDay()
            day_1 = now - 1 * BDay()
            day_0 = now - 0 * BDay() # Add current day_0

            days  = [day_5.day, day_4.day, day_3.day, day_2.day, day_1.day, day_0.day]
            months = [day_5.month, day_4.month, day_3.month, day_2.month, day_1.month, day_0.month]
            years = [day_5.year, day_4.year, day_3.year, day_2.year, day_1.year, day_0.year]
            days  = [str(d) for d in days]
            months  = [str(ms) for ms in months]
            years  = [str(ys) for ys in years]
                
            # for day in days:
                for (day, month, year) in itertools.izip(days, months, years):
                    try:

# closest biz day in the past

            def closest_business_day_in_past(date=None):
                if date is None:
                    date = dt.datetime.today()
                return date + BDay(1) - BDay(1)


            date = pd.datetime.strptime(pd.datetime.now().strftime('%Y%m%d'),'%Y%m%d')  
                    - pd.offsets.BDay(1)




# -------------------------------------------------------------------------------------------------------------    
# utc to local

import datetime

def utc_to_local(self, utc_dt):
    utc_dt = datetime.strptime(utc_dt, "%Y-%m-%d %H:%M:%S")
    local = utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
    return local.strftime("%Y-%m-%d %H:%M:%S")

def utc_to_local(utc_dt):
    local = utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
    return local.strftime("%Y-%m-%d %H:%M:%S")

utc_dt = datetime.strptime("2018-11-01 01:45:00", "%Y-%m-%d %H:%M:%S")
print(utc_to_local(utc_dt))

# -------------------------------------------------------------------------------------------------------------    
# string to datetime -> calculate -> date string for d days ago
# today is August 13, 10 days ago means August 3

def days_ago(d, start_date=None):
    if start_date==None:
        date = datetime.datetime.today() - datetime.timedelta(days=d)
    else:
        date = str_to_date(start_date) - datetime.timedelta(days=d)
    return date.strftime("%Y-%m-%d")

def str_to_date(dt):
    year, month, day = (int(x) for x in dt.split('-'))    
    return datetime.date(year, month, day)

# -------------------------------------------------------------------------------------------------------------
# several time ranges during trading session

start_trading_day = datetime.date(2017, 6, 8)
end_trading_day = datetime.date(2017, 6, 20) #end_trading_day
# end_trading_day  = datetime.date.today()
trading_days = []
while start_trading_day <= end_trading_day:
    trading_days.append(start_trading_day)
    start_trading_day += datetime.timedelta(days=1)

for trading_d in trading_days:
    start_datetime1 = datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 9, 0, 0)
    end_datetime1 =   datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 10, 15, 0)
    start_datetime2 = datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 10, 30, 0)
    end_datetime2 =   datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 11, 30, 0)
    start_datetime3 = datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 13, 30, 0)
    end_datetime3 =   datetime.datetime(trading_d.year, trading_d.month, trading_d.day, 15, 0, 0)
    time_query = {
        '$or':[{'snapshot_time':{'$gte':start_datetime1, '$lte':end_datetime1}},
               {'snapshot_time':{'$gte':start_datetime2, '$lte':end_datetime2}},
               {'snapshot_time':{'$gte':start_datetime3, '$lte':end_datetime3}}
               ]}

# -------------------------------------------------------------------------------------------------------------
# mask for selecting only trading hours

def trading_start(d):
    mkt_open = dt.datetime( int(year), int(month), int(d), 9, 30 )
    return mkt_open

def trading_end(d):
    mkt_close = dt.datetime( int(year), int(month), int(d), 16, 00 )
    return mkt_close

def trading_hours(data):
    test = []
    for d in days:
        dat = data[ ( data.index > trading_start(d) ) & ( data.index < trading_end(d) ) ]
        test.append( dat )
    return test

# -------------------------------------------------------------------------------------------------------------
# check if market is open

import logging
import os
import inspect
import sys
import time
import pytz
from enum import Enum
from datetime import datetime
from govuk_bank_holidays.bank_holidays import BankHolidays

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from Utility.Utils import Utils


class TimeAmount(Enum):
    """Types of amount of time to wait for
    """

    SECONDS = 0
    NEXT_MARKET_OPENING = 1


class TimeProvider:
    """Class that handle functions dependents on actual time
    such as wait, sleep or compute date/time operations
    """

    def __init__(self):
        logging.debug("TimeProvider __init__")

    def is_market_open(self, timezone):
        """
        Return True if the market is open, false otherwise
            - **timezone**: string representing the timezone
        """
        tz = pytz.timezone(timezone)
        now_time = datetime.now(tz=tz).strftime("%H:%M")
        return BankHolidays().is_work_day(datetime.now(tz=tz)) and Utils.is_between(
            str(now_time), ("07:55", "16:35")
        )

    def get_seconds_to_market_opening(self, from_time):
        """Return the amount of seconds from now to the next market opening,
        taking into account UK bank holidays and weekends"""
        today_opening = datetime(
            year=from_time.year,
            month=from_time.month,
            day=from_time.day,
            hour=8,
            minute=0,
            second=0,
            microsecond=0,
        )

        if from_time < today_opening and BankHolidays().is_work_day(from_time.date()):
            nextMarketOpening = today_opening
        else:
            # Get next working day
            nextWorkDate = BankHolidays().get_next_work_day(date=from_time.date())
            nextMarketOpening = datetime(
                year=nextWorkDate.year,
                month=nextWorkDate.month,
                day=nextWorkDate.day,
                hour=8,
                minute=0,
                second=0,
                microsecond=0,
            )
        # Calculate the delta from from_time to the next market opening
        return (nextMarketOpening - from_time).total_seconds()

    def wait_for(self, time_amount_type, amount=-1):
        """Wait for the specified amount of time.
        An TimeAmount type can be specified
        """
        if time_amount_type is TimeAmount.NEXT_MARKET_OPENING:
            amount = self.get_seconds_to_market_opening(datetime.now())
        elif time_amount_type is TimeAmount.SECONDS:
            if amount < 0:
                raise ValueError("Invalid amount of time to wait for")
        logging.info("Wait for {0:.2f} hours...".format(amount / 3600))
        time.sleep(amount)




# -------------------------------------------------------------------------------------------------------------
# creating datetime object from zip

with zipfile.ZipFile(self.fname) as zf:
    with zf.open(zf.namelist()[0]) as infile:
        header = infile.readline()
        datestr, record_count = header.split(b':')
        self.month = int(datestr[2:4])
        self.day = int(datestr[4:6])
        self.year = int(datestr[6:10])
        utc_base_time = datetime(self.year, self.month, self.day)
        self.base_time = timezone('US/Eastern').localize(utc_base_time).timestamp()



# -------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------
# creating url which contains date

    date = pd.datetime.today() - pd.offsets.BDay(1)
    date=date.date()
    datestr = date.strftime('%Y%m%d')
    daystr=str(date.day)
    monthstr=str(date.month)
    yearstr=str(date.year)
    
    
    url='http://www.netfonds.no/quotes/exchange.php?'  
    url=url+'exchange=%s'
    url=url+'&at_day=' + daystr
    url=url+'&at_month=' +monthstr
    url=url+'&at_year=' +yearstr
    url=url+'&format=csv'



# -------------------------------------------------------------------------------------------------------------
# prepare date string

    def prepare_date_strings(date):

        date_yr = date.year.__str__()
        date_mth = date.month.__str__()
        date_day = date.day.__str__()

        if len(date_mth) == 1: date_mth = '0' + date_mth
        if len(date_day) == 1: date_day = '0' + date_day
        
        return date_yr, date_mth, date_day

# --------------------------------------------------------------------------------------------------------------
# date to string 
current_date = Date(2015, 7, 24) # create date object
two_days_later = current_date + 2 # => Date(2015, 7, 26) 
str(two_days_later) # => 2015-07-26
current_date + '1M' # => Date(2015, 8, 24)
current_date + Period('1M') # same with previous line # => Date(2015, 8, 24)

current_date.strftime("%Y%m%d") # => '20150724'
Date.strptime('20160115', '%Y%m%d') # => Date(2016, 1, 15)
Date.strptime('2016-01-15', '%Y-%m-%d') # => Date(2016, 1, 15)
Date.from_datetime(datetime.datetime(2015, 7, 24)) # => Date(2015, 7, 24)

# ---------------------------------------------------------------------------------------------------------------
# string to date

pd.to_datetime(pd.Series(["Jul 31, 2017","2010-10-01","2016/10/10","2014.06.10"]))
pd.to_datetime(pd.Series(["11 Jul 2018","13.04.2015","30/12/2011"]),dayfirst=True)
# providing a format could increase speed of conversion significantly
pd.to_datetime(pd.Series(["12-11-2010 01:56","11-01-2012 22:10","28-02-2013 14:59"]), format='%d-%m-%Y %H:%M')

import market_calendars as mcal
cal_sse = mcal.get_calendar('China.SSE') # create chinese shanghai stock exchange calendar
cal_sse.adjust_date('20130131') # => datetime.datetime(2013, 1, 31, 0, 0)
cal_sse.adjust_date('20130131', return_string=True) # => '2013-01-31'
cal_sse.adjust_date('2017/10/01') # => datetime.datetime(2017, 10, 9, 0, 0)
cal_sse.adjust_date('2017/10/01', convention=2) # => datetime.datetime(2017, 9, 29, 0, 0)



# ---------------------------------------------------------------------------------------------------


# https://github.com/alexanu/market_calendars
# Chinese and US trading calendars with date math utilities 
# based on pandas_market_calendar
# Speed is achieved via Cython

import market_calendars as mcal
from market_calendars.core import Date, Period, Calendar, Schedule

cal_sse = mcal.get_calendar('China.SSE') # create chinese shanghai stock exchange calendar
cal_nyse = mcal.get_calendar('NYSE') # create nyse calendar
cal_sse.name, cal_sse.tz # return name and time zone => ('China.SSE', <DstTzInfo 'Asia/Shanghai' LMT+8:06:00 STD>)

cal_sse.holidays('2018-09-20', '2018-10-10') # return holidays in datetime format
cal_sse.holidays('2018-09-20', '2018-10-10', return_string=True) # return holidays in string format
cal_sse.holidays('2018-09-20', '2018-10-10', return_string=True, include_weekends=False) # return holidays excluding weekends
cal_sse.biz_days('2015-05-20', '2015-06-01') # return biz days in datetime format

cal_sse.is_biz_day('2014-09-22'), cal_sse.is_biz_day('20140130') # => (True, True)
cal_sse.is_holiday('2016-10-01'), cal_sse.is_holiday('2014/9/21') # => (True, True)
cal_sse.is_weekend('2014-01-25'), cal_sse.is_weekend('2011/12/31') # => (True, True)
cal_sse.is_end_of_month('2011-12-30'), cal_sse.is_end_of_month('20120131') # => (True, True)



cal_sse.advance_date('2017-04-27', '2b') # => datetime.datetime(2017, 5, 2, 0, 0)
cal_sse.advance_date('20170427', '2b', return_string=True) # => '2017-05-02'
cal_sse.advance_date('20170427', '1w', return_string=True) # => '2017-05-04'
cal_sse.advance_date('20170427', '1m', return_string=True) # => '2017-05-31'
cal_sse.advance_date('20170427', '-1m', return_string=True) # => '2017-03-27'

cal_sse.schedule('2018-01-05', '2018-02-01', '1w', return_string=True, date_generation_rule=2) # => ['2018-01-05', '2018-01-12', '2018-01-19', '2018-01-26', '2018-02-01']


# -------------------------------------------------------------------------------------------------------------
# reading time stamp
def read_hhmmss(field: str) -> int:
    """Read a HH:MM:SS field and return us since midnight."""
    if field != "":
        hour = int(field[0:2])
        minute = int(field[3:5])
        second = int(field[6:8])
        return 1000000 * ((3600 * hour) + (60 * minute) + second)
    else:
        return 0


def read_hhmmssmil(field: str) -> int:
    """Read a HH:MM:SS:MILL field and return us since midnight."""
    if field != "":
        hour = int(field[0:2])
        minute = int(field[3:5])
        second = int(field[6:8])
        msecs = int(field[9:])
        return ((1000000 * ((3600 * hour) + (60 * minute) + second)) +
                (1000 * msecs))
    else:
        return 0


def read_mmddccyy(field: str) -> np.datetime64:
    """Read a MM-DD-CCYY field and return a np.datetime64('D') type."""
    if field != "":
        month = int(field[0:2])
        day = int(field[3:5])
        year = int(field[6:10])
        return np.datetime64(
            datetime.date(year=year, month=month, day=day), 'D')
    else:
        return np.datetime64(datetime.date(year=1, month=1, day=1), 'D')


def read_ccyymmdd(field: str) -> np.datetime64:
    """Read a CCYYMMDD field and return a np.datetime64('D') type."""
    if field != "":
        year = int(field[0:4])
        month = int(field[4:6])
        day = int(field[6:8])
        return np.datetime64(
            datetime.date(year=year, month=month, day=day), 'D')
    else:
        return np.datetime64(datetime.date(year=1, month=1, day=1), 'D')


def read_timestamp_msg(dt_tm: str) -> Tuple[np.datetime64, int]:
    """Read a CCYYMMDD HH:MM:SS field."""
    if dt_tm != "":
        (date_str, time_str) = dt_tm.split(' ')
        dt = read_ccyymmdd(date_str)
        tm = read_hhmmss(time_str)
        return dt, tm
    else:
        return np.datetime64(datetime.date(year=1, month=1, day=1), 'D'), 0


def read_hist_news_timestamp(dt_tm: str) -> Tuple[np.datetime64, int]:
    """Read a news story time"""
    if dt_tm != "":
        date_str = dt_tm[0:8]
        time_str = dt_tm[8:14]
        dt = read_ccyymmdd(date_str)
        tm = read_hhmmss_no_colon(time_str)
        return dt, tm
    else:
        return np.datetime64(datetime.date(year=1, month=1, day=1), 'D'), 0

#-------------------------------------------------------------------------------------

def us_since_midnight_to_time(us_dt: Union[int, np.int64]) -> datetime.time:
    """Convert us since midnight to datetime.time with rounding."""
    us = np.int64(us_dt)
    assert us >= 0
    assert us <= 86400000000
    microsecond = us % 1000000
    secs_since_midnight = np.floor(us / 1000000.0)
    hour = np.floor(secs_since_midnight / 3600)
    minute = np.floor((secs_since_midnight - (hour * 3600)) / 60)
    second = secs_since_midnight - (hour * 3600) - (minute * 60)
    return datetime.time(hour=int(hour),
                         minute=int(minute),
                         second=int(second),
                         microsecond=int(microsecond))
                         
def date_us_to_datetime(dt64: np.datetime64,
                        tm_int: Union[int, np.datetime64]) -> datetime.datetime:
    """Convert a np.datetime64('D') and us_since midnight to datetime"""
    dt = datetime64_to_date(dt64)
    tm = us_since_midnight_to_time(tm_int)
    return datetime.datetime(year=dt.year, month=dt.month, day=dt.day,
                             hour=tm.hour, minute=tm.minute, second=tm.second,
                             microsecond=tm.microsecond)



# -------------------------------------------------------------------------------------------------------------
# transform to the valid data

# value "09/2007" to date 2007-09-01. 
# value "2006" to date 2016-01-01.

def parse_thisdate(text: str) -> datetime.date:
    parts = text.split('/')
    if len(parts) == 2:
        return datetime.date(int(parts[1]), int(parts[0]), 1)
    elif len(parts) == 1:
        return datetime.date(int(parts[0]), 1, 1)
    else:
        assert False, 'Unknown date format'


# -------------------------------------------------------------------------------------------------------------
# Pandas with @staticmethod







start=pd.Timestamp("2018-01-06 00:00:00")
pd.date_range(start, periods=10,freq="2h20min")

# Business days and biz hours
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




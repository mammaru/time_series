import os
import pandas as pd
from ..core.series import TimeSeries

base = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.normpath(os.path.join(base, 'data'))

def exchange(normalization=True):
    filename = data_path + '/exchange.dat'
    print 'Loading data from: ' + filename
    df = pd.read_table(filename, index_col="datetime")
    df.index = pd.to_datetime(df.index) # convert index into datetime
    #hourly = df.resample("H", how="mean") # hourly
    daily = df.resample("D", how="mean") # daily
    price = daily.ix[:, daily.columns.map(lambda x: x.endswith("PRICE"))]
    volume = daily.ix[:, daily.columns.map(lambda x: x.endswith("VOLUME"))]
    return {"price": TimeSeries(price), "volume": TimeSeries(volume)}

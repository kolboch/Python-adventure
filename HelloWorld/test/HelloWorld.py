import math
import pandas as pd
import quandl

quandl.ApiConfig.api_key = '_nwroafrR4Sp74d2Ugpp'
df = quandl.get_table("WIKI/PRICES")                                        #df from data frame

df = df[['adj_open','adj_high','adj_low','adj_close','adj_volume']]         #from all data getting only specified part

df['HL_PCT'] = (df['adj_high'] - df['adj_low']) / df['adj_close'] * 100         #percent change ( HL_PCT - high low _ percentage )
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100    #daily change

df = df[['adj_close', 'HL_PCT', 'PCT_change', 'adj_volume']] #again changin data columns we need

print(df.head(7))       #returns specified number of rows from head of list ( default 5 ? )

forecast_col = 'adj_close'      # for forecast column
df.fillna(-99999, inplace=True)        # fill NA ( non available ) data

forecast_out = int(math.ceil(0.001*len(df)))

df['label'] = df[forecast_col]

print(df.head(5))

df['label'] = df[forecast_col].shift(-forecast_out) # shift 'moves' data up/down in our case when -value it will go up
                                        # so basically nr of column + shift argument -> new column nr
df.dropna(inplace=True) # for dropping data with non available data ( such data occurs after shifting )
print(df.head(5))
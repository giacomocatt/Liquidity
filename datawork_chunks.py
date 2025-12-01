import pandas as pd
import numpy as np
import os

# Set working directory
os.chdir("C:/Users/giaco/Desktop/phd/Current Projects/Liquidity/Data")

# Read data
df = pd.read_csv('data_trace.csv')
issue = pd.read_csv('Issues.csv')

df['datetime'] = pd.to_datetime(df['trd_exctn_dt'] + ' ' + df['trd_exctn_tm'])

chunks = np.array_split(df, 20)

def compute_trades_last_30_days(chunk):
    chunk = chunk.set_index('datetime')
    chunk['trade_count'] = 1
    chunk = chunk.sort_values(by=['cusip_id', 'datetime'])
    chunk['trades_last_30_days'] = (
        chunk.groupby('cusip_id')['trade_count']
             .apply(lambda x: x.rolling('30D').sum())
             .reset_index(level=0, drop=True)
    )
    return chunk

def add_turnover(chunk):
    chunk = chunk.merge(issue, 
                        left_on='cusip_id', 
                        right_on = 'COMPLETE_CUSIP', 
                        how='left')
    chunk['turnover'] = 100* chunk['entrd_vol_qt'] / (chunk['AMOUNT_OUTSTANDING'] + 1000*chunk['OFFERING_AMT'])
    return chunk

def aggregate_daily_stats(chunk):
    chunk['trd_exctn_dt'] = pd.to_datetime(chunk['trd_exctn_dt'])
    daily = (
    chunk.groupby(['cusip_id', 'trd_exctn_dt'])
         .agg(
             price=('rptd_pr', 'mean'),
             volume=('entrd_vol_qt', 'mean'),
             yield_=('yld_pt', 'mean'),
             turnover=('turnover', 'mean'),
             n_trades=('trades_last_30_days', 'mean')
         )
         .reset_index()
)
    return daily

processed_chunks = []
for chunk in chunks:
    # Combine date and time into a single datetime column
    #chunk['datetime'] = pd.to_datetime(chunk['date'] + ' ' + chunk['time'])
    # Sort and compute rolling window
    chunk = chunk.sort_values(by=['cusip_id', 'datetime'])
    chunk = compute_trades_last_30_days(chunk)
    chunk = add_turnover(chunk)
    aggregated_chunk = aggregate_daily_stats(chunk)

    processed_chunks.append(aggregated_chunk)

daily = pd.concat(processed_chunks, ignore_index=True)
daily = daily.sort_values(by=['cusip_id', 'trd_exctn_dt'])

daily = pd.read_csv('daily_aggregated.csv')
maturity = pd.read_csv('Issues.csv')[['COMPLETE_CUSIP', 'MATURITY', 'ISSUER_CUSIP']]
daily = pd.merge(daily, maturity, 
                 left_on = 'cusip_id', 
                 right_on = 'COMPLETE_CUSIP', 
                 how='left'
                 )
daily['MATURITY'] = pd.to_datetime(daily['MATURITY'])
daily['trd_exctn_dt'] = pd.to_datetime(daily['trd_exctn_dt'])
daily['ttm'] = (daily['MATURITY'] - daily['trd_exctn_dt']).dt.days/30
daily = daily.drop(columns = ['MATURITY', 'COMPLETE_CUSIP'])

rating = pd.read_csv('Ratings.csv')
rating = rating[rating['RATING_TYPE'] == 'FR']
rating = rating[['RATING_DATE', 'RATING', 'COMPLETE_CUSIP']]
rating = rating[rating['COMPLETE_CUSIP'].isin(set(daily['cusip_id']))]
rating['RATING_DATE'] = pd.to_datetime(rating['RATING_DATE'])
rating = rating.sort_values(by=['COMPLETE_CUSIP', 'RATING_DATE'])
rating['end_date'] = rating.groupby('COMPLETE_CUSIP')['RATING_DATE'].shift(-1)
rating['end_date'].fillna(pd.Timestamp('2025-01-01'), inplace=True)
from datetime import timedelta
def expand_dates(row):
    return pd.DataFrame({
        'cusip_id': row['COMPLETE_CUSIP'],
        'date': pd.date_range(start=row['RATING_DATE'], end=row['end_date'] - timedelta(days=1)),
        'rating': row['RATING']
    })
ratings = pd.concat(rating.apply(expand_dates, axis=1).to_list(), ignore_index=True)

daily = pd.merge(daily, ratings, 
                 left_on = ['cusip_id', 'trd_exctn_dt'], 
                 right_on = ['cusip_id', 'date'], 
                 how='left'
                 )

daily.to_csv('daily_aggregated.csv', index=False)

def pivot_side(chunk):
    chunk = (
    chunk.groupby(['cusip_id', 'trd_exctn_dt', 'rpt_side_cd', 'cntra_mp_id'])
         .agg(
             price=('rptd_pr', 'mean')
         )
         .reset_index()
)
    chunk_wide = chunk.pivot(index=['trd_exctn_dt', 'cusip_id'], 
                             columns=['rpt_side_cd', 'cntra_mp_id'], 
                             values=['price'])
    chunk_wide.columns = ['_'.join(map(str, col)).strip() for col in chunk_wide.columns]
    chunk_wide.reset_index(inplace=True)
    return chunk_wide

def spreads(chunk_wide):
    chunk_wide['ba_c'] = 200 * abs(chunk_wide['price_S_C'] - chunk_wide['price_B_C']) / (chunk_wide['price_S_C'] + chunk_wide['price_B_C'])
    chunk_wide['ba_d'] = 200 * abs(chunk_wide['price_S_D'] - chunk_wide['price_B_D']) / (chunk_wide['price_S_D'] + chunk_wide['price_B_D'])
    chunk_wide = chunk_wide.drop(columns =['price_S_D','price_B_D','price_S_D','price_B_D','price_S_C','price_B_C','price_S_C','price_B_C'])
    return chunk_wide

processed_chunks_wide = []
for chunk in chunks:
    chunk['datetime'] = pd.to_datetime(chunk['trd_exctn_dt'] + ' ' + chunk['trd_exctn_tm'])
    chunk_wide = pivot_side(chunk)
    chunk_wide = spreads(chunk_wide)

    processed_chunks_wide.append(chunk_wide)

daily_wide = pd.concat(processed_chunks_wide, ignore_index=True)

def compute_roll(chunk):
    chunk_roll = chunk.groupby(['cusip_id', 'trd_exctn_dt'])['log_return'].apply(roll_gamma_estimator).reset_index()
    return chunk_roll

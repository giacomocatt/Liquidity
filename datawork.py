import pandas as pd
import numpy as np
import os

# Set working directory
os.chdir("C:/Users/giaco/Desktop/phd/Current Projects/Liquidity/Data")

# Read data
df = pd.read_csv('data_trace.csv')
additions = pd.read_csv('Issue_&_rating.csv')[['OFFERING_AMT', 'AMOUNT_OUTSTANDING', 'COMPLETE_CUSIP', 'PRINCIPAL_AMT', 'MATURITY']]

# Select relevant columns from df_origin
# Merge datasets
data = pd.merge(df, additions, left_on='cusip_id', right_on='COMPLETE_CUSIP.x', how='left')

# Filter out rows with missing 'trd_exctn_dt'
data = data[data['trd_exctn_dt'].notna()]

# Add new columns
data['ttm'] = data['ttm'] / 30
data['turnover'] = data['entrd_vol_qt'] / (data['OFFERING_AMT'] + data['AMOUNT_OUTSTANDING'])

# Index(['Unnamed: 0', 'cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'rptd_pr',
#       'entrd_vol_qt', 'yld_pt', 'rpt_side_cd', 'cntra_mp_id',
#       'ISSUER_CUSIP.x', 'MATURITY.x', 'OFFERING_DATE.x', 'OFFERING_PRICE_x',
#       'OFFERING_YIELD', 'COUPON', 'RATING_TYPE', 'RATING_x', 'rtng', 'ttm',
#       'ttm_agg', 'OFFERING_AMT', 'AMOUNT_OUTSTANDING', 'COMPLETE_CUSIP.x',
#       'PRINCIPAL_AMT', 'OFFERING_PRICE_y', 'RATING_y', 'turnover', 'amihud'],
#      dtype='object')

# Group by and calculate weighted means and averages
data = pd.read_csv('data_all.csv')
data = data.sort_values(by=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm'])
data['log_price'] = np.log(data['rptd_pr'])
data['log_return'] = data.groupby(['cusip_id', 'trd_exctn_dt'])['log_price'].diff()
#data['amihud'] = data['log_return'].abs() / data['entrd_vol_qt']
def roll_gamma_estimator(returns):
    if len(returns) > 1:
        cov_matrix = np.cov(returns[:-1], returns[1:])
        return -cov_matrix[0, 1] if cov_matrix.shape == (2, 2) else np.nan
    return np.nan
roll_gamma_df = data.groupby(['cusip_id', 'trd_exctn_dt'])['log_return'].apply(roll_gamma_estimator).reset_index()

daily = (
    data.groupby(['trd_exctn_dt', 'cusip_id', 'rpt_side_cd', 'cntra_mp_id', 'ttm'])
    .agg(
        price=('rptd_pr', lambda x: np.average(x, weights=data.loc[x.index, 'entrd_vol_qt'])),
        vol=('entrd_vol_qt', 'sum'),
        yield_=('yld_pt', lambda x: np.average(x, weights=data.loc[x.index, 'entrd_vol_qt'])),
        turnover=('turnover', 'mean'),
        amihud=('amihud', 'mean'),
        impact=('impact', 'mean')
    )
    .reset_index()
)

daily.to_csv("daily.csv", index=False)

data = pd.read_csv('data_trace.csv')
df['trd_exctn_tm'] = pd.to_datetime(df['trd_exctn_tm'])
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])
df['cusip_id'] = df['cusip_id'].astype('category')
df['entrd_vol_qt'] = df['entrd_vol_qt'].astype('float32')
df['rptd_pr'] = df['rptd_pr'].astype('float32')

df = df.sort_values(by=['cusip_id', 'trd_exctn_dt'])
# Add a trade count column (each row represents one trade)
df['trade_count'] = 1
df['trade_count'] = df['trade_count'].astype('int32')

# Group by security and apply a rolling window of 30 days
df['trd_nmbr'] = (
    df.groupby('cusip_id')
      .apply(lambda x: x.set_index('trd_exctn_dt')['trade_count'].rolling(window=30).sum())
      .reset_index(level=0, drop=True)
)

data = pd.merge(df, additions, left_on='cusip_id', right_on='COMPLETE_CUSIP.x', how='left')
data['turnover'] = data['entrd_vol_qt'] / (data['OFFERING_AMT'] + data['AMOUNT_OUTSTANDING'])

daily = (
    data.groupby(['trd_exctn_dt', 'cusip_id', 'rpt_side_cd', 'cntra_mp_id'])
    .agg(
        price=('rptd_pr', lambda x: np.average(x, weights=data.loc[x.index, 'entrd_vol_qt'])),
        vol=('entrd_vol_qt', 'sum'),
        ttm = ('ttm', 'mean'),
        yield_=('yld_pt', lambda x: np.average(x, weights=data.loc[x.index, 'entrd_vol_qt'])),
        turnover=('turnover', 'mean'),
        n_trades=('trd_nmbr', 'sum')
    )
    .reset_index()
)

from PyCurve.nelson_siegel import NelsonSiegel  # Ensure you have a library for Nelson-Siegel curves
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from PyCurve.curve import Curve

# Load and preprocess yield curve data
y_curve = pd.read_csv('yield_curve.csv')
y_curve['Date'] = pd.to_datetime(y_curve['Date'], format='%m/%d/%y')
y_curve = y_curve.drop(columns = ["4 Mo"])

mat_treas = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])  # Maturities in years

# Convert yield curve to time-series indexed by Date
y_curve.set_index('Date', inplace=True)

# Perform Nelson-Siegel estimation (adjust this based on available libraries or implementation)
ns = NelsonSiegel(0.3,0.4,12,1)
nscoeffs = []
for date,row in y_curve.iterrows():
    rates = row.values
    arr = np.array([rates, mat_treas])
    cleaned_arr = arr[:, ~np.isnan(arr).any(axis=0)]
    curve = Curve(cleaned_arr[1], cleaned_arr[0])
    calibr = ns.calibrate(curve)
    nscoeffs.append([date, calibr.x[0], calibr.x[1], calibr.x[2], calibr.x[3]])

# Convert to DataFrame
nscoeffs = pd.DataFrame(nscoeffs, columns=['Date', 'beta0', 'beta1', 'beta2', 'lambda'])

nscoeffs = pd.read_csv('nscoeffs.csv') # RIGHT ONE!!!
nscoeffs['Date'] = pd.to_datetime(nscoeffs['Date'])
#nscoeffs.set_index('Date', inplace=True)

# Define the risk-free rate function based on Nelson-Siegel parameters
#def rf(t_date, tau):
#    t_date = pd.to_datetime(t_date)
#    ns = nscoeffs.loc[t_date]
#    tau = tau / (12)  # Convert to years
#    beta0, beta1, beta2, lambda_ = ns['beta0'], ns['beta1'], ns['beta2'], ns['lambda']
#    term1 = beta1 * (1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)
#    term2 = beta2 * ((1 - np.exp(-lambda_ * tau)) / (lambda_ * tau) - np.exp(-lambda_ * tau))
#    return beta0 + term1 + term2

def rf(row):
    tau = row['ttm'] / (12)  # Convert to years
    beta0, beta1, beta2, lambda_ = row['beta0'], row['beta1'], row['beta2'], row['lambda']
    term1 = beta1 * (1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)
    term2 = beta2 * ((1 - np.exp(-lambda_ * tau)) / (lambda_ * tau) - np.exp(-lambda_ * tau))
    return beta0 + term1 + term2

df = pd.read_csv('daily_aggregated.csv')
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])
#df['trd_exctn_dt'] = df['trd_exctn_dt'].dt.date
#missing_elements = set(df["trd_exctn_dt"]) - set(nscoeffs.index)
#df = df[~df["trd_exctn_dt"].isin(missing_elements)]
#tqdm.pandas()  # Progress bar for apply
#df['rf'] = df.progress_apply(lambda row: rf(row['trd_exctn_dt'], row['ttm']), axis=1)

df = pd.merge(df, nscoeffs, left_on='trd_exctn_dt', right_on = 'Date', how='left')
df['rf'] = df.apply(rf, axis=1)
df['cs'] = df['yield_'] - df['rf']
df = df.drop(columns=['Date', 'beta0', 'beta1', 'beta2', 'lambda'])

df.to_csv('daily_aggregated.csv', index=False)

# Load datasets
df = pd.read_csv('finratios_2019_2024.csv')
bonds = pd.read_csv('daily.csv')

# Create ISSUER_CUSIP and date-related fields
df['ISSUER_CUSIP'] = df['cusip'].str[:6]
df['public_date'] = pd.to_datetime(df['public_date'])
df['quarter'] = df['public_date'].dt.quarter
df['year'] = df['public_date'].dt.year

bonds['trd_exctn_dt'] = pd.to_datetime(bonds['trd_exctn_dt'])
bonds['quarter'] = bonds['trd_exctn_dt'].dt.quarter
bonds['year'] = bonds['trd_exctn_dt'].dt.year

# Aggregate financial ratios by quarter and year
df_agg = (
    df.groupby(['year', 'quarter', 'cusip', 'ISSUER_CUSIP'])
    .agg({
        'opmbd': 'mean',
        'roa': 'mean',
        'cash_ratio': 'mean',
        'totdebt_invcap': 'mean',
        'debt_ebitda': 'mean',
        'cash_debt': 'mean',
        'debt_capital': 'mean',
        'intcov_ratio': 'mean',
        'curr_ratio': 'mean'
    })
    .reset_index()
)

# Add month for the quarter start and create date field
df_agg['month'] = df_agg['quarter'].replace({1: '01', 2: '04', 3: '07', 4: '10'})
df_agg['date'] = pd.to_datetime(df_agg[['year', 'month']].assign(day=1).astype(str).agg('-'.join, axis=1))


# Group data by ISSUER_CUSIP.x for creating lags
groups = df_agg.groupby('ISSUER_CUSIP')

# Apply lags
def add_lags(group):
    group = group.sort_values('date')
    group['roa_lag'] = group['roa'].shift(1)
    group['debt_capital_lag'] = group['debt_capital'].shift(1)
    group['opmbd_lag'] = group['opmbd'].shift(1)
    group['cash_debt_lag'] = group['cash_debt'].shift(1)
    group['cash_ratio_lag'] = group['cash_ratio'].shift(1)
    group['intcov_ratio_lag'] = group['intcov_ratio'].shift(1)
    group['curr_ratio_lag'] = group['curr_ratio'].shift(1)
    group['roa_lag_2'] = group['roa'].shift(2)
    group['debt_capital_lag_2'] = group['debt_capital'].shift(2)
    group['opmbd_lag_2'] = group['opmbd'].shift(2)
    group['cash_debt_lag_2'] = group['cash_debt'].shift(2)
    group['cash_ratio_lag_2'] = group['cash_ratio'].shift(2)
    group['intcov_ratio_lag_2'] = group['intcov_ratio'].shift(2)
    group['curr_ratio_lag_2'] = group['curr_ratio'].shift(2)
    group['roa_lag_3'] = group['roa'].shift(3)
    group['debt_capital_lag_3'] = group['debt_capital'].shift(3)
    group['opmbd_lag_3'] = group['opmbd'].shift(3)
    group['cash_debt_lag_3'] = group['cash_debt'].shift(3)
    group['cash_ratio_lag_3'] = group['cash_ratio'].shift(3)
    group['intcov_ratio_lag_3'] = group['intcov_ratio'].shift(3)
    group['curr_ratio_lag_3'] = group['curr_ratio'].shift(3)
    group['roa_lag_4'] = group['roa'].shift(4)
    group['debt_capital_lag_4'] = group['debt_capital'].shift(4)
    group['opmbd_lag_4'] = group['opmbd'].shift(4)
    group['cash_ratio_lag_4'] = group['cash_ratio'].shift(4)
    group['cash_debt_lag_4'] = group['cash_debt'].shift(4)
    group['intcov_ratio_lag_4'] = group['intcov_ratio'].shift(4)
    group['curr_ratio_lag_4'] = group['curr_ratio'].shift(4)
    return group

df_lagged = groups.apply(add_lags).reset_index(drop=True)

# Merge bond data with financial ratios
df_or = pd.read_csv('Issue_&_rating.csv')  # Replace with actual file name
df_or = df_or[['COMPLETE_CUSIP.x', 'ISSUER_CUSIP.x']]

bonds = pd.merge(bonds, df_or, left_on='cusip_id', right_on='COMPLETE_CUSIP.x')
bonds['quarter'] = bonds['trd_exctn_dt'].dt.quarter
bonds['year'] = bonds['trd_exctn_dt'].dt.year
# Final merge with lagged data
data = pd.merge(bonds, df_lagged, on=['quarter', 'year', 'ISSUER_CUSIP.x'], how='left')


df = pd.read_csv('daily_aggregated.csv')
yc = pd.read_csv('yield_curve.csv')[['Date', '1 Mo', '3 Mo', '10 Yr']]

# Preprocess yield curve data
yc.columns = ['Date', 'short_r', 'short', 'long']
yc['slope'] = yc['long'] - yc['short']
yc = yc[['Date', 'short_r', 'slope']]
yc['Date'] = pd.to_datetime(yc['Date'], format='%m/%d/%y')

# Join yield curve data with the main dataframe
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])
df = pd.merge(df, yc, left_on='trd_exctn_dt', right_on='Date', how='left')
df = df[df['cusip_id'].notna()]

stocks = pd.read_csv('crsp_liq.csv')
stocks['date'] = pd.to_datetime(stocks['date'])
crsp_trace = pd.read_csv('crsp_trace.csv')
df = pd.merge(
    df, crsp_trace,
    how='left',
    left_on=['cusip_id'],  # Columns in Dataset 1
    right_on=['CUSIP']     # Corresponding columns in Dataset 2
)

df = pd.merge(
    df, stocks,
    how='left',
    left_on=['trd_exctn_dt', 'PERMNO'],  # Columns in Dataset 1
    right_on=['date', 'PERMNO']     # Corresponding columns in Dataset 2
)

vix = pd.read_csv('vix_daily.csv')[['Date', 'vixo']]
vix['Date']=pd.to_datetime(vix['Date'])
df = pd.merge(df, vix, left_on = 'trd_exctn_dt', right_on = 'Date', how='left')

spread = pd.read_csv('ICE_BofA_HY_spread.csv')
spread['observation_date'] = pd.to_datetime(spread['observation_date'])
df = pd.merge(df, spread, left_on = 'trd_exctn_dt', right_on = 'observation_date', how='left')
df = df.rename(columns={'OPENPRC':'stock_pr', 'PERMCO_x':'PERMCO', 'ISSUER_CUSIP':'issuer_cusip', 'vixo':'vix', 'BAMLH0A0HYM2':'ice_bofa_hy_spread'})
df = df.drop(columns=['date_x', 'Date_x', 'CUSIP', 'date_y', 'PERMCO_y', 'Date_y', 'observation_date'])

# Moving averages
df = df.sort_values(by=['cusip_id', 'trd_exctn_dt'])
df['S_ma'] = df.groupby('cusip_id')['S'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['B_ma'] = df.groupby('cusip_id')['B'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['BA'] = 200 * (df['S_ma'] - df['B_ma']) / (df['S_ma'] + df['B_ma'])


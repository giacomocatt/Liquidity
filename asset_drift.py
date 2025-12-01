import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm

import os
os.chdir("C:/Users/giaco/Desktop/phd/Current Projects/Liquidity/Data")

cusips = pd.read_csv('company_codes.csv')
assets = pd.read_csv('assets.csv')
assets = assets.loc[assets['LPERMNO'].isin(cusips['PERMNO'])]

# Sort by firm and date
df = assets[['LPERMNO', 'LPERMCO', 'datadate', 'fyearq','actq','atq']].sort_values(['LPERMCO', 'datadate'])
df['datadate'] = pd.to_datetime(df['datadate'])
df['log_A'] = np.log(df['atq']*1e9)

tau = 1/4  

# Function to compute negative log-likelihood
def log_likelihood(params, Y_tau, tau):
    mu, log_sigma_sq = params
    sigma_sq = np.exp(log_sigma_sq)
    
    # Log-likelihood function
    logL = -0.5 * np.sum(
        np.log(2 * np.pi * sigma_sq * tau) +
        (Y_tau - (mu - 0.5 * sigma_sq) * tau)**2 / (sigma_sq * tau)
    )
    
    return -logL  # Negative log-likelihood (for minimization)

# Function to estimate MLE for each firm
def estimate_gbm_params(sub_df):
    sub_df['Y_tau'] = sub_df['log_A'].diff().dropna()  # Compute log-differences
    
    sub_df_valid = sub_df.dropna(subset=['Y_tau'])
    if sub_df_valid.empty:
        return None
    initial_params = [0, 0]
    result = minimize(
        log_likelihood, initial_params,
        args=(sub_df_valid['Y_tau'], tau),
        method='L-BFGS-B',
        bounds=[(None, None), (None, None)]  # Ensure sigma^2 is positive
    )

    # Extract MLE estimates
    mu_hat, log_sigma_sq_hat = result.x
    sigma_hat = np.sqrt(np.exp(log_sigma_sq_hat))

    return pd.Series({
        'LPERMNO': sub_df['LPERMNO'].iloc[0], 
        'mu': mu_hat,
        'sigma': sigma_hat
    })

# Apply regression to each firm
results = df.groupby('LPERMNO').apply(estimate_gbm_params).reset_index(drop=True)
df = df.merge(results, on='LPERMNO', how='left')

debt_at = pd.read_csv('debt_at.csv')
debt_at['qdate'] = pd.to_datetime(debt_at['qdate'])
debt_at = debt_at.drop(columns = ['gvkey', 'adate', 'public_date'])
debt_at['quarter'] = debt_at['qdate'].dt.quarter
debt_at['year'] = debt_at['qdate'].dt.year
ddf = pd.merge(df, debt_at, left_on = ['quarter', 'year', 'PERMNO'],
              right_on = ['quarter', 'year', 'permno'], how = 'left')
ddf = ddf.dropna(subset=['debt_at', 'cusip_id'])
#df.to_csv('data_regression_debtat.csv')

def KMV_spread(row):
    try:
        A_B = 1 / row['debt_at']
        ttm = row['ttm']/12
        sigma = row['sigma']
        mu = row['mu']
        rf = np.log(1+row['rf']/100)

        if ttm <= 0 or sigma <= 0 or A_B <= 0:
            return np.nan  # or raise an error or return 0
        
        d1 = (np.log(A_B) + (rf + 0.5 * sigma**2) * ttm) / (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)
        arg = np.exp(rf*ttm) * A_B * norm.cdf(-d1) + norm.cdf(d2)
        if arg <= 0:
            return np.nan
        else:
            return -1 / ttm * np.log(arg)
    except Exception:
        return np.nan

ddf['kmv_spread'] = ddf.apply(KMV_spread, axis=1)
dddf = ddf.dropna(subset='kmv_spread')
dddf = dddf[(dddf['kmv_spread'] >= np.percentile(dddf['kmv_spread'], 1)) & 
        (dddf['kmv_spread'] <= np.percentile(dddf['kmv_spread'], 99))]
plt.hist(dddf['rf'], bins = 100)
plt.show()
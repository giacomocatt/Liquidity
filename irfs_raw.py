import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels import OLS
from linearmodels.iv.model import IV2SLS
from linearmodels.panel.model import PanelOLS, RandomEffects, PooledOLS
from matplotlib import cm

# Set working directory (adjust to your environment)
import os
os.chdir("C:/Users/giaco/Desktop/phd/Current Projects/Liquidity/Data")

# Load data

df = pd.read_csv('data_regression_clean.csv')
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])
df = df[(df['turnover'] >= 0) & (df['turnover'] <= np.percentile(df['turnover'], 95))]
df = df[(df['cs'] >= 0) & (df['cs'] <= np.percentile(df['cs'], 99))]
#df = df[(df['ba_c'] >= 0) & (df['ba_c'] <= 25)]
mp_df = pd.read_excel('MPshocksAcosta.xlsx', sheet_name='shocks')
mp_df['fomc'] = pd.to_datetime(mp_df['fomc'])
mp_shocks = mp_df[['fomc', 'target', 'path', 'ns']]

def categorize_rating(rating):
    if rating in ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
                                'A1', 'A2','A3', 'Aa1', 'Aa2', 'Aa3', 'Aaa']:
        result = 'AAA-A'  # Investment Grade
    else:
        if rating in ['BBB+', 'BBB', 'BBB-',
                      'Baa1', 'Baa2', 'Baa3']:
            result = 'BBB'
        else:
            result = 'speculative'
    return result

df['investment_grade'] = df['rating'].apply(categorize_rating)
df = pd.merge(df, mp_df, left_on = 'trd_exctn_dt', right_on = 'fomc', how='left')
df['turn_shock'] = df['ff.shock.0']*df['turnover']
df['nr_shock'] = df['ff.shock.0']*df['n_trades']
df['ba_shock'] = df['ff.shock.0']*df['ba_c']

# Create IVs
df['month'] = df['trd_exctn_dt'].dt.month
iv = (
    df.groupby(['month', 'year', 'investment_grade'])
    .agg({
        #'ba_c': 'mean',
        #'turnover': 'mean',
        #'yield_': 'mean',
        'turn_shock': 'mean',
        'nr_shock': 'mean'
    })
    .reset_index()
)

iv['date'] = pd.to_datetime(iv['year'].astype(str) + '-' + iv['month'].astype(str)#.map({'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'})
                            + '-01')
iv = iv.sort_values('date')
iv = iv.rename(columns={'ba_c':'ba_avg', 'turnover':'turnover_avg', 'yield_':'yield_avg'})
#iv[['ba_avg_lag', 'turnover_avg_lag', 'yield_avg_lag']] = iv[['ba_avg', 'turnover_avg', 'yield_avg']].shift(1)

# Merge IVs with the main dataframe
df = pd.merge(df, iv, on=['month', 'year', 'investment_grade'], how='left')

df = df[df['ff.shock.0'].notna()]

def iv_regression(df):
    df = df.set_index(['cusip_id', 'trd_exctn_dt'])
    dep = df[['cs']]
    endog = df[['turnover','turn_shock_x']]
    exog = sm.add_constant(df[['ff.shock.0',
                            'ttm', 
                            #'opmbd', 'roa', 'debt_capital', 'debt_ebitda','cash_ratio', 'intcov_ratio', 'curr_ratio', 
               #'roa_lag', 'debt_capital_lag', 'opmbd_lag', 'cash_ratio_lag', 'intcov_ratio_lag','curr_ratio_lag', 
               #'roa_lag_2', 'debt_capital_lag_2', 'opmbd_lag_2','cash_ratio_lag_2', 'intcov_ratio_lag_2', 'curr_ratio_lag_2',
               #'roa_lag_3', 'debt_capital_lag_3', 'opmbd_lag_3', 'cash_ratio_lag_3','intcov_ratio_lag_3', 'curr_ratio_lag_3',
               #'roa_lag_4', 'debt_capital_lag_4', 'opmbd_lag_4', 'cash_ratio_lag_4','intcov_ratio_lag_4', 'curr_ratio_lag_4', 
               'short_r', 'slope', 'vix', 'ice_bofa_hy_spread', 'investment_grade']])
    iv = df[['n_trades','nr_shock_x']]
    model = IV2SLS(dep, exog, endog, iv).fit()
    return model

regr = iv_regression(df)
print(regr)

df.set_index('trd_exctn_dt', inplace=True)
max_horizon = 30  # Maximum horizon in days for the IRF
response_var = 'cs'  # Name of the response variable
control_vars = ['ttm', 'opmbd_lag', 'roa_lag', 'debt_capital_lag',
                'stock_pr', 'short_r', 'slope', 'vix', 'ice_bofa_hy_spread', 'investment_grade']  # Aggregate control variables
endog_vars = ['turnover', 'turn_shock_x']
iv_vars = ['n_trades', 'nr_shock_x']
# Initialize a dictionary to store coefficients and standard errors
irf_results = {'horizon': [], 'beta0': [], 'upper0': [], 'lower0': [],
                              'beta1': [], 'upper1': [], 'lower1': []}

# Compute the IRF for each horizon h
for h in range(max_horizon + 1):
    # Shift the response variable by h days to get credit_spread_{i,t+h}
    df[f'{response_var}_shifted'] = df[response_var].shift(-h)
    
    # Drop rows with missing values (caused by the shift)
    regression_data = df.dropna(subset=[f'{response_var}_shifted', 'ff.shock.0'])
    
    # Define the independent variables (shock and controls)
    X = regression_data[['ff.shock.0'] + control_vars]
    X = sm.add_constant(X)  # Add a constant term for the regression
    endog = regression_data[endog_vars]
    iv = regression_data[iv_vars]
    
    # Define the dependent variable
    y = regression_data[f'{response_var}_shifted']
    
    # Run the OLS regression
    model = IV2SLS(y, X, endog, iv)
    results = model.fit()
    
    # Extract the coefficient and standard error for the shock variable
    beta0_h = results.params['ff.shock.0']
    ci0_h = results.conf_int().loc['ff.shock.0']
    beta1_h = results.params['turn_shock_x']
    ci1_h = results.conf_int().loc['turn_shock_x']
    
    # Store the results
    irf_results['horizon'].append(h)
    irf_results['beta0'].append(beta0_h)
    irf_results['lower0'].append(ci0_h[0])
    irf_results['upper0'].append(ci0_h[1])
    irf_results['beta1'].append(beta1_h)
    irf_results['lower1'].append(ci1_h[0])
    irf_results['upper1'].append(ci1_h[1])

# Convert results to a DataFrame
irf_df = pd.DataFrame(irf_results)

# Plot the IRF
plt.figure(figsize=(10, 6))
plt.plot(irf_df['horizon'], irf_df['beta0'], label='IRF', color='blue')
plt.fill_between(irf_df['horizon'], irf_df['lower0'], irf_df['upper0'], color='blue', alpha=0.2, label='Confidence Interval')
plt.plot(irf_df['horizon'], irf_df['beta1'], label='Interaction', color='orange')
plt.fill_between(irf_df['horizon'], irf_df['lower1'], irf_df['upper1'], color='orange', alpha=0.2, label='Confidence Interval')
plt.axhline(0, color='red', linestyle='--')
plt.title('Impulse Response Function to Monetary Policy Shocks')
plt.xlabel('Horizon (days)')
plt.ylabel(r'$\beta0_h$ (Response)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df['liquidity_portfolio'], edges = pd.qcut(df['turnover'], q=2, labels=['low', 'high'], retbins=True)
df['portfolio'] = df['investment_grade'].astype(str) + '_' + df['liquidity_portfolio'].astype(str)
portfolios = df['portfolio'].unique()
liquidity_portfolios = df['portfolio'].unique()
control_vars = ['ttm', 
                #'roa_lag', 'debt_capital_lag', 'opmbd_lag', 'cash_ratio_lag', 'intcov_ratio_lag','curr_ratio_lag', 
                'short_r', 'slope', 'vix', 'ice_bofa_hy_spread', 'investment_grade']  # Aggregate control variables

irf_results = {'portfolio': [],'horizon': [], 'beta': [], 'upper': [], 'lower': []}

for portfolio in portfolios:
    portfolio_data = df[df['portfolio'] == portfolio]
    for h in range(max_horizon + 1):
        portfolio_data[f'{response_var}_shifted'] = portfolio_data[response_var].shift(-h)
        regression_data = portfolio_data.dropna(subset=[f'{response_var}_shifted', 'ff.shock.0'])

        # Define the independent variables (shock and controls)
        X = regression_data[['ff.shock.0'] + control_vars]
        X = sm.add_constant(X)
        
        # Define the dependent variable
        y = regression_data[f'{response_var}_shifted']
        
        # Run the OLS regression
        model = OLS(y, X)
        results = model.fit()
        
        # Extract the coefficient and standard error for the shock variable
        beta_h = results.params['ff.shock.0']
        ci_h = results.conf_int().loc['ff.shock.0']
        
        # Store the results
        irf_results['portfolio'].append(portfolio)
        irf_results['horizon'].append(h)
        irf_results['beta'].append(beta_h)
        irf_results['lower'].append(ci_h[0])
        irf_results['upper'].append(ci_h[1])
        
irf_df = pd.DataFrame(irf_results)
colors = cm.Blues(np.linspace(0.3, 1, len(portfolios)))
# Plot IRFs for each portfolio
plt.figure(figsize=(14, 8))
portfolio_irf = irf_df[irf_df['portfolio'] == 'medium']
plt.plot(portfolio_irf['horizon'], portfolio_irf['beta'], label='Above 66 percentile', color=colors[1])
plt.fill_between(portfolio_irf['horizon'], portfolio_irf['lower'], portfolio_irf['upper'], color=colors[1], alpha=0.2)

portfolio_irf = irf_df[irf_df['portfolio'] == 'low'].iloc[0:31]
plt.plot(portfolio_irf['horizon'], portfolio_irf['beta'], label='33 to 66 percentile', color=colors[2])
plt.fill_between(portfolio_irf['horizon'], portfolio_irf['lower'], portfolio_irf['upper'], color=colors[2], alpha=0.2)

# Customize the plot
plt.axhline(0, color='red', linestyle='--')
plt.title('Impulse Response Functions by Liquidity Level')
plt.xlabel('Horizon (days)')
plt.ylabel(r'$\beta_h$ (Response)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
for i, portfolio in enumerate(portfolios):
    portfolio_irf = irf_df[irf_df['portfolio'] == portfolio]
    plt.plot(portfolio_irf['horizon'], portfolio_irf['beta'], label=f'Portfolio: {portfolio}', color=colors[i])
    #plt.fill_between(portfolio_irf['horizon'], portfolio_irf['lower'], portfolio_irf['upper'], color=colors[i], alpha=0.2)

# Customize the plot
plt.axhline(0, color='red', linestyle='--')
plt.title('Impulse Response Functions by Portfolio')
plt.xlabel('Horizon (days)')
plt.ylabel(r'$\beta_h$ (Response)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

rating_class = df['investment_grade'].unique()
shock_var = 'target'
response_var = 'turnover'
control_vars = [#'ttm', 
                #'roa_lag', 'debt_capital_lag', 'opmbd_lag', 'cash_ratio_lag', 'intcov_ratio_lag','curr_ratio_lag', 
                #'short_r', 'slope', 'vix', 
                #'ice_bofa_hy_spread', 
                'investment_grade']  # Aggregate control variables
max_horizon = 30
irf_results = {'horizon': [], 'beta': [], 'upper': [], 'lower': []}

for h in range(max_horizon + 1):
    df[f'{response_var}_shifted'] = df[response_var].shift(-h)
    regression_data = df.dropna(subset=[f'{response_var}_shifted', shock_var])
    X = regression_data[[shock_var] + control_vars]
    X = sm.add_constant(X)
    y = regression_data[f'{response_var}_shifted']
    model = OLS(y, X)
    results = model.fit()
        
    beta_h = results.params[shock_var]
    ci_h = results.conf_int().loc[shock_var]
        
        # Store the results
    irf_results['horizon'].append(h)
    irf_results['beta'].append(beta_h)
    irf_results['lower'].append(ci_h[0])
    irf_results['upper'].append(ci_h[1])
        
irf_df = pd.DataFrame(irf_results)
plt.plot(irf_df['horizon'], irf_df['beta'], label='IRF', color='blue')
plt.fill_between(irf_df['horizon'], irf_df['lower'], irf_df['upper'], color='blue', alpha=0.2)

# Customize the plot
plt.axhline(0, color='red', linestyle='--')
plt.title('Impulse Response Functions of Bid-Ask Spreads')
plt.xlabel('Horizon (days)')
plt.ylabel(r'$\beta_h$ (Response)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

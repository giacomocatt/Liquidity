import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from linearmodels import OLS
from linearmodels.iv.model import IV2SLS
from linearmodels.panel.model import PanelOLS, RandomEffects, PooledOLS
from matplotlib import cm
from scipy.stats import norm
confidence_level = 0.05
# Set working directory (adjust to your environment)
import os
os.chdir("C:/Users/giaco/Desktop/phd/Current Projects/Liquidity/Data")

# Load data

df = pd.read_csv('data_regression_clean.csv')
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])

df = df.drop_duplicates(subset=['cusip_id', 'trd_exctn_dt'])
#df = df.dropna(subset='liq')

df['liquidity_portfolio'], edges = pd.qcut(df['turnover'], q=2, labels=['low', 'high'], retbins=True)
df['ig_hy'] = df['investment_grade'].apply(lambda x: 'HY' if x=='speculative' else 'IG')
df['portfolio'] = df['ig_hy'].astype(str) + '_' + df['liquidity_portfolio'].astype(str)
portfolios = df['portfolio'].unique()

def oneway_demean(df, cols, id_col):
    """
    Demean multiple columns by a single fixed effect (id_col).
    Returns a DataFrame with the same column names.
    """
    # group means for all variables at once
    mean_id = df.groupby(id_col)[cols].transform("mean")
    
    # grand means
    grand_mean = df[cols].mean()
    
    # demeaned variables
    return df[cols] - mean_id + grand_mean

def twoway_demean(df, cols, id_col, time_col):
    """
    Efficient two-way fixed effects demeaning for multiple columns.
    cols: list of column names to demean
    id_col: entity identifier (e.g., cusip_id)
    time_col: time identifier (e.g., trd_exctn_dt)
    Returns a DataFrame with demeaned variables.
    """

    # Mean by entity (CUSIP)
    mean_id = df.groupby(id_col)[cols].transform("mean")

    # Mean by time
    mean_time = df.groupby(time_col)[cols].transform("mean")

    # Grand mean
    grand_mean = df[cols].mean()

    # Two-way demean
    return df[cols] - mean_id - mean_time + grand_mean


def local_projections(data, response_var, control_vars, policy_var, shock_var, max_horizon, confidence_level):
    z_score = norm.ppf(1-confidence_level/2) 
    irf_results = {'horizon': [], 'beta': [], 'upper': [], 'lower': [],
                       'beta_HY_high': [], 'upper_HY_high': [], 'lower_HY_high': [],
                       'beta_HY_low': [], 'upper_HY_low': [], 'lower_HY_low': [],
                       'beta_IG_high': [], 'upper_IG_high': [], 'lower_IG_high': []}
    for h in range(max_horizon + 1):
        data[f'{response_var}_shifted'] = data[response_var].shift(-h)
        regression_data = data.dropna(subset=[f'{response_var}_shifted', shock_var] + control_vars)
        regression_data['HY_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'HY')).astype(int)
        regression_data['HY_low'] = ((regression_data['liquidity_portfolio'] == 'low')*(regression_data['ig_hy'] == 'HY')).astype(int)
        regression_data['IG_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'IG')).astype(int)
        regression_data['shock_HY_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'HY')*regression_data[shock_var])
        regression_data['shock_HY_low'] = ((regression_data['liquidity_portfolio'] == 'low')*(regression_data['ig_hy'] == 'HY')*regression_data[shock_var])
        regression_data['shock_IG_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'IG')*regression_data[shock_var])
        regression_data['ffr_HY_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'HY')*regression_data[policy_var])
        regression_data['ffr_HY_low'] = ((regression_data['liquidity_portfolio'] == 'low')*(regression_data['ig_hy'] == 'HY')*regression_data[policy_var])
        regression_data['ffr_IG_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'IG')*regression_data[policy_var])
        
        # Define the independent variables (shock and controls)
        x_endog = twoway_demean(regression_data, 
                               [policy_var,'ffr_HY_high','ffr_HY_low','ffr_IG_high'], 
                            "cusip_id", 'trd_exctn_dt')
        exog_vars = ['HY_high','HY_low','IG_high'] + control_vars
        x_exog = twoway_demean(regression_data, 
                               exog_vars, 
                            "cusip_id", 'trd_exctn_dt')
        instr_vars = ['shock_HY_high','shock_HY_low','shock_IG_high', shock_var]
        z = twoway_demean(regression_data,
                          instr_vars, 
                            "cusip_id", 'trd_exctn_dt')
        
        # Define the dependent variable
        y = twoway_demean(regression_data, [f'{response_var}_shifted'], "cusip_id", 'trd_exctn_dt')
        
        # Run the OLS regression
        model =  IV2SLS(
            dependent=y,
            exog=x_exog,
            endog=x_endog,
            instruments=z
                )
        results = model.fit(cov_type='robust')
        
        # Extract the coefficient and standard error for the shock variable
        beta = results.params[policy_var]
        beta_HY_high = results.params['ffr_HY_high']
        beta_HY_low = results.params['ffr_HY_low']
        beta_IG_high = results.params['ffr_IG_high']
        
        cov = results.cov
       
       # Sum 1: beta + beta_HY_high
        beta_sum_HY_high = beta + beta_HY_high
        var_sum_HY_high = cov.loc[policy_var, policy_var] + cov.loc['ffr_HY_high', 'ffr_HY_high'] + 2 * cov.loc[policy_var, 'ffr_HY_high']
        ci_sum_HY_high = z_score * np.sqrt(var_sum_HY_high)
        
        beta_sum_HY_low = beta + beta_HY_low
        var_sum_HY_low = (
            cov.loc[policy_var, policy_var] +
            cov.loc['ffr_HY_low', 'ffr_HY_low'] +
            2 * cov.loc[policy_var, 'ffr_HY_low']
        )
        ci_sum_HY_low = z_score * np.sqrt(var_sum_HY_low)
        
        beta_sum_IG_high = beta + beta_IG_high
        var_sum_IG_high = (
            cov.loc[policy_var, policy_var] +
            cov.loc['ffr_IG_high', 'ffr_IG_high'] +
            2 * cov.loc[policy_var, 'ffr_IG_high']
        )
        ci_sum_IG_high = z_score * np.sqrt(var_sum_IG_high)
       
       
       # CI for base beta
        se_beta = np.sqrt(cov.loc[policy_var, policy_var])
        ci_base = z_score * se_beta
        
        
        # Store the results
        irf_results['horizon'].append(h)
        irf_results['beta'].append(beta)
        irf_results['beta_HY_high'].append(beta_sum_HY_high)
        irf_results['beta_HY_low'].append(beta_sum_HY_low)
        irf_results['beta_IG_high'].append(beta_sum_IG_high)
        irf_results['lower'].append(beta - ci_base)
        irf_results['upper'].append(beta + ci_base)
        irf_results['lower_HY_high'].append(beta_sum_HY_high - ci_sum_HY_high)
        irf_results['upper_HY_high'].append(beta_sum_HY_high + ci_sum_HY_high)
        irf_results['lower_HY_low'].append(beta_sum_HY_low - ci_sum_HY_low)
        irf_results['upper_HY_low'].append(beta_sum_HY_low + ci_sum_HY_low)
        irf_results['lower_IG_high'].append(beta_sum_IG_high - ci_sum_IG_high)
        irf_results['upper_IG_high'].append(beta_sum_IG_high + ci_sum_IG_high)
    irf_df = pd.DataFrame(irf_results)
    return irf_df

response_var = 'yield_'
policy_var = 'ffr'

control_vars = []#['ttm', 
               # 'roa', 'debt_capital', 'opmbd', 'cash_ratio', 'intcov_ratio','curr_ratio', 
                #'short_r', 
                #'slope', 
                #'vix'
                #, 'stock_pr'
                #,'ice_bofa_hy_spread'
                #]
max_horizon = 90

shocks = ['GSS_target', 'GSS_path', 'ns']
irf_multi = {}
for shock in shocks:
    #current_controls = control_vars[:] if shock == 'GSS_target' else control_vars + ['stock_pr']
    irf_df = local_projections(df, response_var, control_vars, policy_var, shock, max_horizon, confidence_level)
    irf_multi[shock] = irf_df
    irf_multi[shock]['shock'] = shock
    
irf_df = pd.concat(irf_multi.values(), ignore_index=True)

mp_df = pd.read_excel('MPshocksAcosta.xlsx', sheet_name='shocks')
sd_shocks_df = mp_df.iloc[:, 1:].std().reset_index().rename(columns={'index': 'shock', 0: 'value'})

shocks = [('GSS_target', 'GSS target'),
          ('GSS_path', 'GSS path')
          ,('ns', 'NS')]
colors = np.vstack([
    cm.Blues(np.linspace(0.7, 1, 2)),
    cm.Reds(np.linspace(0.7, 1, 2))
])
fig, axes = plt.subplots(1, len(shocks), figsize=(18, 10))

for j, (shock, shock_name) in enumerate(shocks):
        ax = axes[j]
        portfolio_irf = irf_df[irf_df['shock'] == shock]
        portfolio_irf[['beta', 'upper', 'lower', 'beta_HY_high', 'upper_HY_high', 'lower_HY_high',
                       'beta_HY_low', 'upper_HY_low', 'lower_HY_low',
                       'beta_IG_high', 'upper_IG_high', 'lower_IG_high']] *= sd_shocks_df.set_index('shock').at[shock, 'value']
        sns.lineplot(data=portfolio_irf, x="horizon", y="beta", ax=ax, color=colors[0], label='Portfolio: Low-IG')
        sns.lineplot(data=portfolio_irf, x="horizon", y="beta_IG_high", ax=ax, color=colors[1], label='Portfolio: High-IG')
        sns.lineplot(data=portfolio_irf, x="horizon", y="beta_HY_low", ax=ax, color=colors[2], label='Portfolio: Low-HY')
        sns.lineplot(data=portfolio_irf, x="horizon", y="beta_HY_high", ax=ax, color=colors[3], label='Portfolio: High-HY')
        ax.fill_between(portfolio_irf["horizon"], portfolio_irf["lower"], portfolio_irf["upper"], color=colors[0], alpha=0.2)
        ax.fill_between(portfolio_irf["horizon"], portfolio_irf["lower_IG_high"], portfolio_irf["upper_IG_high"], color=colors[1], alpha=0.2)
        ax.fill_between(portfolio_irf["horizon"], portfolio_irf["lower_HY_low"], portfolio_irf["upper_HY_low"], color=colors[2], alpha=0.2)
        ax.fill_between(portfolio_irf["horizon"], portfolio_irf["lower_HY_high"], portfolio_irf["upper_HY_high"], color=colors[3], alpha=0.2)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f'Impulse Responses of Credit Spreads by Portfolio to {shock_name}')
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('Response in p.p.')
        ax.legend()
        ax.grid(True)
        
plt.tight_layout()
plt.show()


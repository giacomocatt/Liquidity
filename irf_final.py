import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
mp_df = pd.read_excel('MPshocksAcosta.xlsx', sheet_name='shocks')
sd_shocks_df = mp_df.iloc[:, 1:].std().reset_index().rename(columns={'index': 'shock', 0: 'value'})
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
confidence_level = 0.05

#PART 1

df = df.drop_duplicates(subset=['cusip_id', 'trd_exctn_dt'])
#df = df.dropna(subset='liq')
df = df.sort_values(["cusip_id", "trd_exctn_dt"])

# rolling median over the past 30 days (entity-specific)
df["turnover_median_30d"] = (
    df.groupby("cusip_id")["turnover"]
      .rolling(window=30, min_periods=1)
      .median()
      .shift(1)  # ensures only past values are used
      .reset_index(level=0, drop=True)
)

# classify into high vs low turnover
df["liquidity_portfolio"] = np.where(
    df["turnover"] > df["turnover_median_30d"],
    "high",
    "low"
)

#df['liquidity_portfolio'], edges = pd.qcut(df['turnover'], q=2, labels=['low', 'high'], retbins=True)
df['ig_hy'] = df['investment_grade'].apply(lambda x: 'HY' if x=='speculative' else 'IG')
df['portfolio'] = df['ig_hy'].astype(str) + '_' + df['liquidity_portfolio'].astype(str)
portfolios = df['portfolio'].unique()
#df_mp = df_mp[['cusip_id', 'trd_exctn_dt', 'liquidity_portfolio', 'portfolio']]
#df = pd.merge(df, df_mp, on = ['cusip_id', 'trd_exctn_dt'], how = 'left')
def local_projections(data, response_var, control_vars, shock_var, max_horizon, confidence_level):
    z_score = norm.ppf(1-confidence_level/2) 
    irf_results = {'horizon': [], 'beta': [], 'upper': [], 'lower': [],
                       'beta_HY_high': [], 'upper_HY_high': [], 'lower_HY_high': [],
                       'beta_HY_low': [], 'upper_HY_low': [], 'lower_HY_low': [],
                       'beta_IG_high': [], 'upper_IG_high': [], 'lower_IG_high': []}
    for h in range(max_horizon + 1):
        data[f'{response_var}_shifted'] = data[response_var].shift(-h)
        regression_data = data.dropna(subset=[f'{response_var}_shifted', shock_var] + control_vars)
        regression_data = regression_data.set_index(['cusip_id','trd_exctn_dt'])
        regression_data[control_vars] = (
                                regression_data.groupby(level="cusip_id")[control_vars]
                                .shift(1)
                                )
        regression_data['HY_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'HY')).astype(int)
        regression_data['HY_low'] = ((regression_data['liquidity_portfolio'] == 'low')*(regression_data['ig_hy'] == 'HY')).astype(int)
        regression_data['IG_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'IG')).astype(int)
        regression_data['shock_HY_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'HY')*regression_data[shock_var])
        regression_data['shock_HY_low'] = ((regression_data['liquidity_portfolio'] == 'low')*(regression_data['ig_hy'] == 'HY')*regression_data[shock_var])
        regression_data['shock_IG_high'] = ((regression_data['liquidity_portfolio'] == 'high')*(regression_data['ig_hy'] == 'IG')*regression_data[shock_var])
        
        # Define the independent variables (shock and controls)
        X = regression_data[[shock_var] + ['HY_high'] + ['HY_low'] + ['IG_high'] +
                            ['shock_HY_high'] + ['shock_HY_low'] + ['shock_IG_high'] + control_vars]
        #X = sm.add_constant(X)
        
        # Define the dependent variable
        y = regression_data[f'{response_var}_shifted']
        
        # Run the OLS regression
        model = PanelOLS(y, X, entity_effects=True)
        results = model.fit()
        
        # Extract the coefficient and standard error for the shock variable
        beta = results.params[shock_var]
        beta_HY_high = results.params['shock_HY_high']
        beta_HY_low = results.params['shock_HY_low']
        beta_IG_high = results.params['shock_IG_high']
        
        cov = results.cov
       
       # Sum 1: beta + beta_HY_high
        beta_sum_HY_high = beta + beta_HY_high
        var_sum_HY_high = cov.loc[shock_var, shock_var] + cov.loc['shock_HY_high', 'shock_HY_high'] + 2 * cov.loc[shock_var, 'shock_HY_high']
        ci_sum_HY_high = z_score * np.sqrt(var_sum_HY_high)
        
        beta_sum_HY_low = beta + beta_HY_low
        var_sum_HY_low = (
            cov.loc[shock_var, shock_var] +
            cov.loc['shock_HY_low', 'shock_HY_low'] +
            2 * cov.loc[shock_var, 'shock_HY_low']
        )
        ci_sum_HY_low = z_score * np.sqrt(var_sum_HY_low)
        
        beta_sum_IG_high = beta + beta_IG_high
        var_sum_IG_high = (
            cov.loc[shock_var, shock_var] +
            cov.loc['shock_IG_high', 'shock_IG_high'] +
            2 * cov.loc[shock_var, 'shock_IG_high']
        )
        ci_sum_IG_high = z_score * np.sqrt(var_sum_IG_high)
       
       
       # CI for base beta
        se_beta = np.sqrt(cov.loc[shock_var, shock_var])
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
#shock_var = 'GSS_path'
response_vars = ['yield_'
                 , 'cs'
                 , 'price']

control_vars = ['ttm', 
                #'roa', 'debt_capital', 'opmbd', 'cash_ratio', 'intcov_ratio','curr_ratio', 
                'short_r', 
                'slope'
                ,'vix'
                #, 'stock_pr'
                #,'ice_bofa_hy_spread'
                ]
max_horizon = 45

shocks = ['GSS_target', 'GSS_path', 'ns']
irf_multi = {}
for y_var in response_vars:
    for shock in shocks:
        #current_controls = control_vars[:] if shock == 'GSS_target' else control_vars + ['stock_pr']
        irf_df = local_projections(df, y_var, control_vars, shock, max_horizon, confidence_level)
        irf_multi[shock, y_var] = irf_df
        irf_multi[shock, y_var]['shock'] = shock
        irf_multi[shock, y_var]['y_var'] = y_var
    
irf_df = pd.concat(irf_multi.values(), ignore_index=True)

shocks = [('GSS_target', 'GSS target'),
          ('GSS_path', 'GSS path')
          ,('ns', 'NS')]
y_vars = [('yield_', 'Yield'),
          ('cs', 'Credit Spread'),
          ('price', 'Price')]
colors = np.vstack([
    cm.Blues(np.linspace(0.7, 1, 2)),
    cm.Reds(np.linspace(0.7, 1, 2))
])
fig, axes = plt.subplots(len(y_vars), len(shocks), figsize=(18, 10))

for i, (y_var, y_name) in enumerate(y_vars):
    for j, (shock, shock_name) in enumerate(shocks):
            ax = axes[i,j]
            portfolio_irf = irf_df[(irf_df['shock'] == shock) & (irf_df['y_var'] == y_var)]
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
            ax.set_title(f'Impulse Responses of {y_name} by Portfolio to {shock_name}')
            ax.set_xlabel('Horizon (days)')
            ax.set_ylabel('Response in p.p.')
            ax.legend()
            ax.grid(True)
        
plt.tight_layout()
plt.show()

#####################LIQUIDITY IRF###############################

def local_projections(data, response_var, control_vars, shock_var, max_horizon, confidence_level):
    z_score = norm.ppf(1-confidence_level/2) 
    irf_results = {'horizon': [], 'beta': [], 'upper': [], 'lower': [],
                       'beta_risk': [], 'upper_risk': [], 'lower_risk': []}
    #fitted_values = {'horizon': [], 'cusip_id': [], 'trd_exctn_dt': [], 'fitted': []}
    for h in range(max_horizon + 1):
        data[f'{response_var}_shifted'] = data[response_var].shift(-h)
        regression_data = data.dropna(subset=[f'{response_var}_shifted', shock_var])
        regression_data = regression_data.set_index(['cusip_id','trd_exctn_dt'])
        regression_data['risk'] = (regression_data['ig_hy'] == 'HY').astype(int)
        regression_data['shock_risk'] = ((regression_data['ig_hy'] == 'HY')*regression_data[shock_var])
        
        # Define the independent variables (shock and controls)
        X = regression_data[[shock_var] + ['risk'] + ['shock_risk'] + control_vars]
        X = sm.add_constant(X)
        
        # Define the dependent variable
        y = regression_data[f'{response_var}_shifted']
        
        # Run the OLS regression
        model = PanelOLS(y, X, entity_effects=True)
        results = model.fit()
        
        # Extract the coefficient and standard error for the shock variable
        beta = results.params[shock_var]
        beta_risk = results.params['shock_risk']
        
        cov = results.cov
        
        beta_sum_risk = beta + beta_risk
        var_sum_risk = (
            cov.loc[shock_var, shock_var] +
            cov.loc['shock_risk', 'shock_risk'] +
            2 * cov.loc[shock_var, 'shock_risk']
        )
        ci_sum_risk = z_score * np.sqrt(var_sum_risk)
        
        se_beta = np.sqrt(cov.loc[shock_var, shock_var])
        ci_base = z_score * se_beta

        
        # Store the results
        irf_results['horizon'].append(h)
        irf_results['beta'].append(beta)
        irf_results['beta_risk'].append(beta_sum_risk)
        irf_results['lower'].append(beta - ci_base)
        irf_results['upper'].append(beta + ci_base)
        irf_results['lower_risk'].append(beta_sum_risk - ci_sum_risk)
        irf_results['upper_risk'].append(beta_sum_risk + ci_sum_risk)
        
        #fitted_values['horizon'].extend([h] * len(results.fittedvalues))
        #fitted_values['cusip_id'].extend(regression_data['cusip_id'].values)
        #fitted_values['trd_exctn_dt'].extend(regression_data['trd_exctn_dt'].values)
        #fitted_values['fitted'].extend(results.fittedvalues)
    irf_df = pd.DataFrame(irf_results)
    #fitted_values_df = pd.DataFrame(fitted_values)
    return irf_df#, fitted_values_df

liquidty_vars = ['turnover', 'ba_c']
shocks = ['GSS_target', 'GSS_path', 'ns']
irf_multi = {}
fitted_multi = {}
for liquidity in liquidty_vars:
    for shock in shocks:
        irf_df= local_projections(df, liquidity, control_vars, shock, max_horizon, confidence_level)
        irf_multi[liquidity, shock] = irf_df
        irf_multi[liquidity, shock]['shock'] = shock
        irf_multi[liquidity, shock]['liquidity'] = liquidity
        #fitted_multi[liquidity, shock] = fitted_df
        #fitted_multi[liquidity, shock]['shock'] = shock
        #fitted_multi[liquidity, shock]['liquidity'] = liquidity
    
irf_df = pd.concat(irf_multi.values(), ignore_index=True)
#fitted_df = pd.concat(fitted_multi.values(), ignore_index=True)

liquidity_vars = [('turnover', 'Turnover'),
          ('ba_c', 'Bid-Ask spread')]
shocks = [('GSS_target', 'GSS target'),
          ('GSS_path', 'GSS path')
          ,('ns', 'NS')]
fig, axes = plt.subplots(2, len(shocks), figsize=(18, 10))

for i, (liq, liq_name) in enumerate(liquidity_vars):
    for j, (shock, shock_name) in enumerate(shocks):
        ax = axes[i,j]
        portfolio_irf = irf_df[(irf_df['shock'] == shock) & (irf_df['liquidity'] == liq )]
        portfolio_irf[['beta', 'upper', 'lower',
                       'beta_risk', 'upper_risk', 'lower_risk']] *= sd_shocks_df.set_index('shock').at[shock, 'value']
        sns.lineplot(data=portfolio_irf, x="horizon", y="beta", ax=ax, color=colors[0], label='IG')
        sns.lineplot(data=portfolio_irf, x="horizon", y="beta_risk", ax=ax, color=colors[2], label='HY')
        ax.fill_between(portfolio_irf["horizon"], portfolio_irf["lower"], portfolio_irf["upper"], color=colors[0], alpha=0.2)
        ax.fill_between(portfolio_irf["horizon"], portfolio_irf["lower_risk"], portfolio_irf["upper_risk"], color=colors[2], alpha=0.2)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f'Impulse Responses of {liq_name} to {shock_name}')
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('Response in p.p.')
        ax.legend()
        ax.grid(True)
        
plt.tight_layout()
plt.show()

#####################CHANNELS###############################


response_var = 'cs'

def local_projections(data, response_var, liquidity_fitted, control_vars, shock_var, max_horizon, confidence_level):
    z_score = norm.ppf(1-confidence_level/2) 
    irf_results = {'horizon': [], 'beta': [], 'upper': [], 'lower': [],
                   'beta_risk': [], 'upper_risk': [], 'lower_risk': [],
                       'beta_liq': [], 'upper_liq': [], 'lower_liq': [],
                       'beta_liq_risk': [], 'upper_liq_risk': [], 'lower_liq_risk': []}
    for h in range(max_horizon + 1):
        data[f'{response_var}_shifted'] = data[response_var].shift(-h)
        liquidity_fitted_h = liquidity_fitted[liquidity_fitted['horizon'] == h]
        regression_data = data.dropna(subset=[f'{response_var}_shifted', shock_var])
        regression_data['shock_risk'] = ((regression_data['ig_hy'] == 'HY')*regression_data[shock_var])
        regression_data = pd.merge(regression_data, liquidity_fitted_h, on = ['cusip_id', 'trd_exctn_dt'], how='inner')
        regression_data['liq_risk'] = ((regression_data['ig_hy'] == 'HY')*regression_data['fitted'])
        
        # Define the independent variables (shock and controls)
        X = regression_data[[shock_var] + ['shock_risk'] + ['fitted'] + ['liq_risk'] + control_vars]
        X = sm.add_constant(X)
        
        # Define the dependent variable
        y = regression_data[f'{response_var}_shifted']
        
        # Run the OLS regression
        model = OLS(y, X)
        results = model.fit()
        
        # Extract the coefficient and standard error for the shock variable
        beta = results.params[shock_var]
        beta_risk = results.params['shock_risk']
        liq_channel = results.params['fitted']
        liq_risk_channel = results.params['liq_risk']
        
        cov = results.cov_params()
        
        beta_sum_risk = beta + beta_risk
        var_sum_risk = (
            cov.loc[shock_var, shock_var] +
            cov.loc['shock_risk', 'shock_risk'] +
            2 * cov.loc[shock_var, 'shock_risk']
        )
        ci_sum_risk = z_score * np.sqrt(var_sum_risk)
        
        se_beta = np.sqrt(cov.loc[shock_var, shock_var])
        ci_base = z_score * se_beta
        
        se_liq = np.sqrt(cov.loc['fitted', 'fitted'])
        ci_liq = z_score * se_liq
        
        beta_liq_risk = liq_channel + liq_risk_channel
        var_liq_risk = (
            cov.loc['fitted', 'fitted'] +
            cov.loc['liq_risk', 'liq_risk'] +
            2 * cov.loc['fitted', 'liq_risk']
        )
        ci_liq_risk = z_score * np.sqrt(var_liq_risk)

        
        # Store the results
        irf_results['horizon'].append(h)
        irf_results['beta'].append(beta)
        irf_results['beta_risk'].append(beta_sum_risk)
        irf_results['beta_liq'].append(liq_channel)
        irf_results['beta_liq_risk'].append(beta_liq_risk)
        irf_results['lower'].append(beta - ci_base)
        irf_results['upper'].append(beta + ci_base)
        irf_results['lower_risk'].append(beta_sum_risk - ci_sum_risk)
        irf_results['upper_risk'].append(beta_sum_risk + ci_sum_risk)
        irf_results['lower_liq'].append(liq_channel - ci_liq)
        irf_results['upper_liq'].append(liq_channel + ci_liq)
        irf_results['lower_liq_risk'].append(beta_liq_risk - ci_liq_risk)
        irf_results['upper_liq_risk'].append(beta_liq_risk + ci_liq_risk)
    irf_df = pd.DataFrame(irf_results)
    return irf_df

irf_multi = {}
for liquidity in liquidty_vars:
    for shock in shocks:
        liquidity_fitted = fitted_df[(fitted_df['shock'] == shock) & (fitted_df['liquidity'] == liquidity)]
        irf_df = local_projections(df, response_var, fitted_df, control_vars, shock, max_horizon, confidence_level)
        irf_multi[liquidity, shock] = irf_df
        irf_multi[liquidity, shock]['shock'] = shock
        irf_multi[liquidity, shock]['liquidity'] = liquidity
    
irf_df = pd.concat(irf_multi.values(), ignore_index=True)

liquidity_vars = [('turnover', 'Turnover'),
          ('ba_c', 'Bid-Ask spread')]
shocks = [('GSS_target', 'GSS target'),
          ('GSS_path', 'GSS path')
          ,('ns', 'NS')]
fig, axes = plt.subplots(2, len(shocks), figsize=(18, 10))

for i, (liq, liq_name) in enumerate(liquidity_vars):
    for j, (shock, shock_name) in enumerate(shocks):
        ax = axes[i,j]
        portfolio_irf = irf_df[(irf_df['shock'] == shock) & (irf_df['liquidity'] == liq )]
        portfolio_irf[['beta_liq', 'upper_liq', 'lower_liq','beta_liq_risk', 'upper_liq_risk', 'lower_liq_risk']] *= sd_shocks_df.set_index('shock').at[shock, 'value']
        sns.lineplot(data=portfolio_irf, x="horizon", y="beta_liq", ax=ax, color=colors[0], label='IG')
        sns.lineplot(data=portfolio_irf, x="horizon", y="beta_liq_risk", ax=ax, color=colors[2], label='HY')
        ax.fill_between(portfolio_irf["horizon"], portfolio_irf["lower_liq"], portfolio_irf["upper_liq"], color=colors[0], alpha=0.2)
        ax.fill_between(portfolio_irf["horizon"], portfolio_irf["lower_liq_risk"], portfolio_irf["upper_liq_risk"], color=colors[2], alpha=0.2)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f'Liquidity Channel of {liq_name} to {shock_name}')
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('Response in p.p.')
        ax.legend()
        ax.grid(True)
        
plt.tight_layout()
plt.show()

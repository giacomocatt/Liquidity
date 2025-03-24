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
#df = df[(df['turnover'] >= 0) & (df['turnover'] <= np.percentile(df['turnover'], 95))]
#df = df[(df['cs'] >= 0) & (df['cs'] <= np.percentile(df['cs'], 99))]
#df = df[(df['ba_c'] >= 0) & (df['ba_c'] <= 25)]

mp_df = pd.read_excel('MPshocksAcosta.xlsx', sheet_name='shocks')

def local_projections(data, response_var, control_vars, shock_var, max_horizon):
    irf_results = {'horizon': [], 'beta': [], 'upper': [], 'lower': []}
    for h in range(max_horizon + 1):
        data[f'{response_var}_shifted'] = data[response_var].shift(-h)
        regression_data = data.dropna(subset=[f'{response_var}_shifted', shock_var])

        # Define the independent variables (shock and controls)
        X = regression_data[[shock_var] + control_vars]
        X = sm.add_constant(X)
        
        # Define the dependent variable
        y = regression_data[f'{response_var}_shifted']
        
        # Run the OLS regression
        model = OLS(y, X)
        results = model.fit()
        
        # Extract the coefficient and standard error for the shock variable
        beta_h = results.params[shock_var]
        ci_h = results.conf_int().loc[shock_var]
        
        # Store the results
        irf_results['horizon'].append(h)
        irf_results['beta'].append(beta_h)
        irf_results['lower'].append(ci_h[0])
        irf_results['upper'].append(ci_h[1])
    irf_df = pd.DataFrame(irf_results)
    return irf_df

shock_var = 'ns'
response_var = 'log_ba'
control_vars = ['ttm', 
                'roa_lag', 'debt_capital_lag', 'opmbd_lag', 'cash_ratio_lag', 'intcov_ratio_lag','curr_ratio_lag', 
                'short_r', 'slope', 'vix', 
                'ice_bofa_hy_spread', 
                'investment_grade']  # Aggregate control variables
max_horizon = 30

irf_df = local_projections(df, response_var, control_vars, shock_var, max_horizon)
shocks = ['ns', 'GSS_target', 'GSS_path']
df['-log_turn'] = np.log(df['turnover'])
df['log_ba'] = df['ba_c'].apply(lambda x: np.log(x) if x>0 else np.log(0.000000000000000000000001))
response_vars=['-log_turn', 'log_ba']
irf_multi = {}

for var in response_vars:
    for shock in shocks:
        irf_df = local_projections(df, var, control_vars, shock, max_horizon)
        irf_multi[shock, var] = irf_df
        irf_multi[shock, var]['shock'] = shock
        irf_multi[shock, var]['response_var'] = var

irf_combined_df = pd.concat(irf_multi.values(), ignore_index=True)

fig, axes = plt.subplots(len(response_vars), len(shocks), figsize=(14, 10))

# Plot each shock with confidence intervals
for j, var in enumerate(response_vars):
    for i, shock in enumerate(shocks):
            ax = axes[j,i]
            df_shock = irf_combined_df[(irf_combined_df["shock"] == shock) & (irf_combined_df["response_var"] == var)]
            sns.lineplot(data=df_shock, x="horizon", y="beta", ax=ax, color='blue', label="Response")
            ax.axhline(0, color='red', linestyle='--')
            ax.fill_between(df_shock["horizon"], df_shock["lower"], df_shock["upper"], color='blue', alpha=0.2)
            ax.set_title(f"{shock} Shock on {var}")
            ax.set_xlabel("Horizon (Days)")
            ax.set_ylabel(r'$\beta_h$ (Response)')
            ax.legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
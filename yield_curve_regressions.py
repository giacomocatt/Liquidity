import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# Set working directory
os.chdir("C:/Users/giaco/Desktop/phd/Current Projects/Liquidity/Data")


mp_df = pd.read_excel('MPshocksAcosta (1).xlsx', sheet_name='shocks')
mp_df['fomc'] = pd.to_datetime(mp_df['fomc'])
#yield_curve = pd.read_csv('yield_curve_data.csv')
yc = pd.read_csv('yield_curve.csv')
yc['Date'] = pd.to_datetime(yc['Date'], format='%m/%d/%y')
#yield_curve['CALDT'] = pd.to_datetime(yc['Date'], format='%m/%d/%y')
vix = pd.read_csv('vix_daily.csv')[['Date', 'vixo']]
vix['Date']=pd.to_datetime(vix['Date'])
spread = pd.read_csv('ICE_BofA_HY_spread.csv')
spread['observation_date'] = pd.to_datetime(spread['observation_date'])
sp500 = pd.read_csv('sp500_daily.csv')
sp500['caldt'] = pd.to_datetime(sp500['caldt'])


df = pd.merge(yc, mp_df, left_on = 'Date', right_on = 'fomc', how = 'left')
df = pd.merge(df, vix, on = 'Date', how = 'left')
df = pd.merge(df, spread, left_on = 'Date', right_on = 'observation_date', how = 'left')
df = pd.merge(df, sp500, left_on = 'Date', right_on = 'caldt', how = 'left')
df['slope'] = df['10 Yr'] - df['3 Mo']

df = df.drop(columns = ['2 Mo', '4 Mo'])
df_reg = df.dropna()
y = df_reg['1 Mo']
X = sm.add_constant(df_reg['ff.shock.0'])
model = sm.OLS(y, X).fit()
model.summary()

df[['totval', 'usdval', 'spindx']] = np.log(df[['totval', 'usdval', 'spindx']])

maturities = yc.drop(columns=['Date', '2 Mo', '4 Mo']).columns
shocks = mp_df.drop(columns=['fomc']).columns
response_var = '10 Yr'
controls = ['vixo',  'vwretd', 'usdval']
irf_results = {'shock': [],'horizon': [], 'beta': [], 'upper': [], 'lower': []}


for T in maturities:
    for di in shocks:
    # Drop rows with missing values (caused by the shift)
        df[f'{T}_shifted'] = df[T].shift(-10)
        regression_data = df.dropna(subset=[f'{T}_shifted', di])
    
    # Define the independent variables (shock and controls)
        X = regression_data[[di]+controls]
        X = sm.add_constant(X)
    
    # Define the dependent variable
        y = regression_data[f'{T}_shifted']
    
    # Run the OLS regression
        model = sm.OLS(y, X)
        results = model.fit()
    
    # Extract the coefficient and standard error for the shock variable
        beta_h = results.params[di]
        ci_h = results.conf_int().loc[di]
    
    # Store the results
        irf_results['shock'].append(di)
        irf_results['horizon'].append(T)
        irf_results['beta'].append(beta_h)
        irf_results['lower'].append(ci_h[0])
        irf_results['upper'].append(ci_h[1])
    
irf_df = pd.DataFrame(irf_results)

fig, axes = plt.subplots(4,1, figsize=(14, 10))

for k, di in enumerate(shocks):
    df_shock = irf_df[irf_df['shock']==di]
    axes[k].plot(df_shock['horizon'], df_shock['beta'], label='IRF', color='blue')
    axes[k].fill_between(df_shock['horizon'], df_shock['lower'], df_shock['upper'], color='blue', alpha=0.2, label='Confidence Interval')
    axes[k].axhline(0, color='red', linestyle='--')
    axes[k].set_title(f'{di}')
axes[3].set_xlabel('Maturity')
plt.tight_layout()
plt.show()

regression_data = df[['Date', 'slope', '10 Yr', 'ns', 'target', 'path', 'ff.shock.0']+controls].dropna()
y = regression_data['slope']
X = sm.add_constant(regression_data[['path']]) #path
model = sm.OLS(y, X)
results = model.fit()
results.summary()

beta_df = pd.read_csv('results_negative_log_turn.csv')
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
beta_df = pd.merge(beta_df, df, on = ['month', 'year'], how = 'left')
beta_df = beta_df.dropna(subset=['target', 'path'])

controls = ['vix', 'ice_bofa_hy_spread']
y = beta_df['estimate']
X = sm.add_constant(beta_df[['ff.shock.0']+controls])
res = sm.OLS(y, X).fit()
res.summary()

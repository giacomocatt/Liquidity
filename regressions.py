import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels import OLS
from linearmodels.iv.model import IV2SLS
from linearmodels.panel.model import PanelOLS, RandomEffects, PooledOLS
import seaborn as sns

# Set working directory (adjust to your environment)
import os
os.chdir("C:/Users/giaco/Desktop/phd/Current Projects/Liquidity/Data")

# Load data

df = pd.read_csv('data_regression.csv')
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])
#df = df[(df['turnover'] >= 0) & (df['turnover'] <= np.percentile(df['turnover'], 95))]
#df = df[(df['cs'] >= 0) & (df['cs'] <= np.percentile(df['cs'], 99))]
#df = df[(df['ba_c'] >= 0) & (df['ba_c'] <= 25)]
plt.hist(df['cs'], bins = 100)
plt.show()

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

# Filter infrequent trading
#trade_counts = df.groupby(['cusip_id', 'quarter', 'year']).size().reset_index(name='frequency')
#trade_counts = trade_counts[trade_counts['frequency'] > 12]
#df = pd.merge(df, trade_counts, on=['cusip_id', 'quarter', 'year'], how='inner')

# Create IVs
df['month'] = df['trd_exctn_dt'].dt.month

# Add lagged variables
#iv['date'] = pd.to_datetime(iv['year'].astype(str) + '-' + iv['month'].astype(str)#.map({'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'})+ '-01')
#iv = iv.sort_values('date')
#iv = iv.rename(columns={'ba_c':'ba_avg', 'turnover':'turnover_avg', 'yield_':'yield_avg'})
#iv[['ba_avg_lag', 'turnover_avg_lag', 'yield_avg_lag']] = iv[['ba_avg', 'turnover_avg', 'yield_avg']].shift(1)

# Merge IVs with the main dataframe
#df = pd.merge(df, iv, on=['month', 'year', 'investment_grade'], how='left')
#df = df[df['rating'].notna()]

#df['d_cs'] = df["cs"] - df['cs_avg']
#df['d_turnover'] = df['turnover'] - df['turnover_avg']
#df['d_yield'] = df['yield'] -df['yield_avg']

aggregates = (
    df.groupby(['month', 'year'])
    .agg({
        'short_r': 'mean',
        'slope': 'mean',
        'stock_pr': 'mean',
        'vix': 'mean',
        'ice_bofa_hy_spread': 'mean'
    })
    .reset_index()
)
aggregates['time'] = pd.PeriodIndex(year=aggregates['year'], month=aggregates['month'], freq='M')
aggregates['time'] = aggregates['time'].dt.to_timestamp()


# Split data by quarter and year
grouped_data = {key: group for key, group in df.groupby(['month', 'year'])}
liq = 'turnover'
# Perform regression
def panel_regression(df):
    df = df.set_index(['cusip_id', 'trd_exctn_dt'])
    dep = df[['cs']]
    exog = sm.add_constant(df[[liq,
                               'ttm', 
                            'opmbd', 'roa', 'debt_capital', 'debt_ebitda',
               'cash_ratio', 'intcov_ratio', 'curr_ratio', 
               'roa_lag', 'debt_capital_lag', 'opmbd_lag', 'cash_ratio_lag', 'intcov_ratio_lag','curr_ratio_lag', 
               'roa_lag_2', 'debt_capital_lag_2', 'opmbd_lag_2','cash_ratio_lag_2', 'intcov_ratio_lag_2', 'curr_ratio_lag_2',
               #'roa_lag_3', 'debt_capital_lag_3', 'opmbd_lag_3', 'cash_ratio_lag_3','intcov_ratio_lag_3', 'curr_ratio_lag_3',
               #'roa_lag_4', 'debt_capital_lag_4', 'opmbd_lag_4', 'cash_ratio_lag_4','intcov_ratio_lag_4', 'curr_ratio_lag_4', 
               'short_r', 'slope', 'stock_pr', 'vix', 'ice_bofa_hy_spread']])
    model = PanelOLS(dep, exog, entity_effects=False, drop_absorbed=True)
    return model

def iv_regression(df):
    df = df.set_index(['cusip_id', 'trd_exctn_dt'])
    dep = df[['cs']][df[liq]>0]
    endog = df[liq]#-np.log(df[liq][df[liq]>0])
    exog = sm.add_constant(df[['ttm', 
                            'opmbd', 'roa', 'debt_capital', 'debt_ebitda',
               'cash_ratio', 'intcov_ratio', 'curr_ratio', 
               'roa_lag', 'debt_capital_lag', 'opmbd_lag', 'cash_ratio_lag', 'intcov_ratio_lag','curr_ratio_lag', 
               'roa_lag_2', 'debt_capital_lag_2', 'opmbd_lag_2','cash_ratio_lag_2', 'intcov_ratio_lag_2', 'curr_ratio_lag_2',
               #'roa_lag_3', 'debt_capital_lag_3', 'opmbd_lag_3', 'cash_ratio_lag_3','intcov_ratio_lag_3', 'curr_ratio_lag_3',
               #'roa_lag_4', 'debt_capital_lag_4', 'opmbd_lag_4', 'cash_ratio_lag_4','intcov_ratio_lag_4', 'curr_ratio_lag_4', 
               'short_r', 'slope', 'vix', 'ice_bofa_hy_spread', 'investment_grade']][df[liq]>0])
    iv = df[['n_trades']][df[liq]>0]
    model = IV2SLS(dep, exog, endog, iv).fit()
    return model

pooled_models = {}
for key, group in grouped_data.items():
    try:
        pooled_models[key] = iv_regression(group)
    except Exception as e:
        print(f"Error in {key}: {e}")

# Extract results
results = []
for key, model in pooled_models.items():
    conf_int = model.conf_int().loc[liq]
    estimate = model.params[liq]
    r2       = model.rsquared
    results.append({'month': key[0], 'year': key[1], 'estimate': estimate, 'lower': conf_int[0], 'upper': conf_int[1], 'R-sq': r2})

results_df = pd.DataFrame(results)
results_df.sort_values(['year', 'month'], inplace=True)

# Plot results
results_df['month'] = results_df['month'].astype(int)
#results_df['quarter_num'] = results_df['quarter'].str.extract('Q(\d)').astype(int)
results_df['time'] = pd.PeriodIndex(year=results_df['year'], month=results_df['month'], freq='M')
results_df = results_df.sort_values('time')  # Ensure data is sorted by time

# Convert `time` to datetime for plotting
results_df['time'] = results_df['time'].dt.to_timestamp()

significant_events = {
    '2007-07-01': 'Subprime Crisis',
    '2008-09-15': 'Lehman Brothers',
    '2009-03-09': 'Stock Market Bottoms',
    '2010-05-06': 'Flash Crash',
    '2011-08-05': 'U.S. Debt Downgrade',
    '2020-02-02': 'COVID-19',
    #'2020-04-20': 'Oil Prices Turn Negative',
    '2021-01-27': 'GameStop'
}
# Plot the time series with confidence intervals
results_df = results_df[results_df['time'] >= '2007-07-01']
results_df = pd.merge(results_df, aggregates, on = 'time', how ='left')
#yc_inversion = results_df['time'][results_df['slope']*results_df['slope'].shift(1) < 0]
#results_df[['estimate', 'lower', 'upper']] = -results_df[['estimate', 'lower', 'upper']]/100
plt.figure(figsize=(14, 6))
plt.plot(results_df['time'], results_df['estimate'], label='Estimate', color='blue')
plt.fill_between(results_df['time'], results_df['lower'], results_df['upper'], color='blue', alpha=0.2, label='Confidence Interval')
plt.axhline(y=0, color='red', linestyle='-', linewidth=1)

for event_date, event_label in significant_events.items():
    event_date = pd.to_datetime(event_date)
    plt.axvline(event_date, color='red', linestyle='--', linewidth=1)
    # Rotate the label
    plt.text(event_date, results_df['estimate'].max(), 
             event_label, color='black',
             verticalalignment='bottom', horizontalalignment='center', rotation=45, fontsize=9)
# Customize the plot
plt.title("Exposure of Credit Spread to Liquidity Over Time", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel(r"$\beta$", fontsize=12)
plt.legend()
plt.grid(True)
year_ticks = pd.date_range(start='2007-07-01', end='2021-12-31', freq='YS')
plt.xticks(year_ticks, year_ticks.strftime('%Y'), rotation=45)
plt.ylim(results_df['estimate'].min()*1.2, results_df['estimate'].max()*1.3)
plt.tight_layout()

# Show the plot
plt.show()

precovid_df = results_df[(results_df['time'] >= '2009-01-01') & (results_df['time'] <= '2019-12-31')]
postcovid_df = results_df[results_df['time'] > '2019-12-31']
plt.figure(figsize=(12, 6))
ax1 = sns.lineplot(x='time', y='estimate', data=results_df, label='Estimate', color='blue')
plt.fill_between(results_df['time'], results_df['lower'], results_df['upper'], color='blue', alpha=0.2, label='Confidence Interval')

# Customize primary y-axis
ax1.set_ylabel(r'$\beta$', fontsize=12)
ax1.set_xlabel('Date')
plt.xticks(rotation=45)

# Create a secondary y-axis and plot the second and third time series
ax2 = ax1.twinx()
#sns.lineplot(x='time', y='short_r', data=resultss_df, label='Yield Curve Level', color='red', ax=ax2)
sns.lineplot(x='time', y='slope', data=results_df, label='10Y-3Mo Treasury Spread', color='green', ax=ax2)

# Customize the secondary y-axis
ax2.set_ylabel('Percentage points', fontsize=12)

# Add legends for both axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Display the plot
plt.title("Liquidity Exposure and Yield Curve Slope", fontsize=14)
year_ticks = pd.date_range(start='2007-06-01', end='2021-12-31', freq='YS')
plt.xticks(year_ticks, year_ticks.strftime('%Y'), rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
ax1 = sns.lineplot(x='time', y='estimate', data=postcovid_df, label='Estimate', color='blue')
plt.fill_between(postcovid_df['time'], postcovid_df['lower'], postcovid_df['upper'], color='blue', alpha=0.2, label='Confidence Interval')

# Customize primary y-axis
ax1.set_ylabel(r'$\beta$', fontsize=12)
ax1.set_xlabel('Date')
plt.xticks(rotation=45)

# Create a secondary y-axis and plot the second and third time series
ax2 = ax1.twinx()
#sns.lineplot(x='time', y='short_r', data=results_df, label='Yield Curve Level', color='red', ax=ax2)
sns.lineplot(x='time', y='slope', data=postcovid_df, label='10Y-3Mo Treasury Spread', color='green', ax=ax2)

# Customize the secondary y-axis
ax2.set_ylabel('Percentage points', fontsize=12)

# Add legends for both axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Display the plot
plt.title("Liquidity Exposure and Yield Curve Slope, Bid-Ask Spreads", fontsize=14)
year_ticks = pd.date_range(start='2020-01-01', end='2021-12-31', freq='YS')
plt.xticks(year_ticks, year_ticks.strftime('%Y'), rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

results_df = pd.read_csv('results_negative_log_turn.csv')
results_df['time'] = pd.to_datetime(results_df['time'])
results_df = results_df[(results_df['time'] >= '2007-06-01' )]
results_df = sm.add_constant(results_df).dropna()
results_df['log_price'] = np.log(results_df['stock_pr'])
check = sm.OLS(results_df['estimate'], 
            results_df[['const','short_r','slope','nasdaq', 'vix']]).fit()
print(check.summary())
from stargazer.stargazer import Stargazer

stargazer = Stargazer([check])
latex_output = stargazer.render_latex()

print(latex_output)

results_df = pd.read_csv('beta_pc1_results.csv')
mp_df = pd.read_excel('MPshocksAcosta.xlsx', sheet_name='shocks')
mp_df['month'] = mp_df['fomc'].apply(lambda x: x.replace(day=1))
results_df['time'] = pd.to_datetime(results_df['time'])

results_df = pd.merge(results_df, mp_df, left_on ='time', right_on = 'month', how = 'left')

results_df['shock_shift'] = results_df['GSS_path'].shift(1)
regr_df = sm.add_constant(results_df).dropna()
check = sm.OLS(regr_df['estimate'], 
            regr_df[['const','short_r', 'slope', 'vix','shock_shift']]).fit()
print(check.summary())
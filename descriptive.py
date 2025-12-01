import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir("C:/Users/giaco/Desktop/phd/Current Projects/Liquidity/Data")

# Load your dataset
df = pd.read_csv("data_regression_clean.csv")

# Ensure the investment grade variable exists (optional, if not created before)
investment_grade_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
                            'A1', 'A2','A3', 'Aa1', 'Aa2', 'Aa3', 'Aaa',
                            'BBB+', 'BBB', 'BBB-',
                            'Baa1', 'Baa2', 'Baa3']
#df['investment_grade'] = df['rating'].apply(lambda x: 'IG' if x in investment_grade_ratings else 'HY')
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
df['log_ba'] = (df[df['ba_c'] > 0]['ba_c'] - np.mean(df[df['ba_c'] > 0]['ba_c']))/np.mean(df[df['ba_c'] > 0]['ba_c'])
df['_log_turnover'] = (df[df['turnover'] > 0]['turnover'] - np.mean(df[df['turnover'] > 0]['turnover']))/np.mean(df[df['turnover'] > 0]['turnover'])

risk_classes = [
    ('AAA-A' , 'royalblue'),
    ('BBB' , 'darkorange'),
    ('speculative', 'yellowgreen')
    ]
variables = [
    ('cs' , 'Credit Spread'),
    ('_log_turnover' , '-log Turnover'),
    ('log_ba', 'log Bid-Ask Spread')
    ]

df_neg = df[(df['cs'] < 0)]
#df = df[(df['n_trades'] <= np.percentile(df['n_trades'],99))]
df = df[(df['ba_c'] >= 0) & (df['ba_c'] <= 25)]
df = df[(df['cs'] >= 0) & (df['cs'] <= np.percentile(df['cs'], 99))]
#plt.hist(np.log(df['ba_c'][df['ba_c']>0]), bins = 100)
#plt.show()
# Set up the figure and axes for the 2x2 grid of distribution plots
fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for i, (risk_class, color) in enumerate(risk_classes):
    for j, (var, varname) in enumerate(variables):
        sns.histplot(df[df['investment_grade'] == risk_class][var], stat="density", ax=axes[i,j], kde=True, color=color)
        axes[i,j].set_title(f'Distribution of {varname} {risk_class}')
        axes[i,j].set_xlabel('p.p.')

plt.tight_layout()
plt.show()

# Convert trd_exctn_dt to trd_exctn_dttime format if necessary
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])

df_summ = (
    df.groupby(['trd_exctn_dt', 'investment_grade'])
    .agg(
        cs=('cs', 'mean'),
        cs_low = ('cs', lambda x: x.mean() - x.std()),
        cs_high = ('cs', lambda x: x.mean() + x.std()),
        _log_turnover=('_log_turnover', 'mean'),
        _log_turnover_low = ('_log_turnover', lambda x: x.mean() - x.std()),
        _log_turnover_high = ('_log_turnover', lambda x: x.mean() + x.std()),
        log_ba =('log_ba', 'mean'),
        log_ba_low = ('log_ba', lambda x: x.mean() - x.std()),
        log_ba_high = ('log_ba', lambda x: x.mean() + x.std()),
    )
    .reset_index()
)

df_summ['trd_exctn_dt'] = pd.to_datetime(df_summ['trd_exctn_dt'])
df_summ = df_summ.sort_values(by=['investment_grade', 'trd_exctn_dt'])
df_summ[['cs', 'cs_low', 'cs_high', 'turnover', 'turnover_low', 'turnover_high', 
         'ba_c', 'ba_c_low', 'ba_c_high']] = df_summ[['cs', 'cs_low', 'cs_high', 
                                                'turnover', 'turnover_low', 'turnover_high', 
                                                'ba_c', 'ba_c_low', 'ba_c_high']].apply(pd.to_numeric, errors='coerce')
df_summ = df_summ.dropna()
df_smooth = (
    df_summ.sort_values(by=['investment_grade', 'trd_exctn_dt'])  # Ensure data is sorted by date
           .groupby('investment_grade')
           .apply(lambda group: group.set_index('trd_exctn_dt')[['cs', 'cs_low', 
                                                                 'cs_high', '_log_turnover', 
                                                                 '_log_turnover_low', '_log_turnover_high', 
                                                                 'log_ba', 'log_ba_low', 'log_ba_high']]
                                    .rolling(window=30, min_periods=1)
                                    .mean()
                                    .reset_index())
           .reset_index()
)



fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for i, (risk_class, color) in enumerate(risk_classes):
    for j, (var, varname) in enumerate(variables):
            sns.lineplot(data=df_smooth[df_smooth['investment_grade'] == risk_class], x='trd_exctn_dt', y= var, ax=axes[i, j], color=color)
            sns.lineplot(data=df_smooth[df_smooth['investment_grade'] == risk_class], x='trd_exctn_dt', y=f'{var}_low', ax=axes[i, j], color=color, linestyle='--')
            sns.lineplot(data=df_smooth[df_smooth['investment_grade'] == risk_class], x='trd_exctn_dt', y=f'{var}_high', ax=axes[i, j], color=color, linestyle='--')
            axes[i,j].set_title(f'Time Series of {varname} {risk_class}')
            axes[i,j].set_xlabel("Time", fontsize=12)
            axes[i,j].set_ylabel('Percentage Points', fontsize=12)

plt.tight_layout()
plt.show()

df = df[(df['ba_c'] >= 0) & (df['ba_c'] <= 25)]
plt.figure(figsize=(8,6))
scatter = plt.scatter(np.log(df['turnover']), df['ba_c'], c=df['riskiness'],
            cmap='viridis', alpha=0.7, edgecolors='k', s= 10)
plt.xlabel("Turnover")
plt.ylabel("Bid-Ask Spread")
plt.grid(True)
plt.show()

from scipy.stats import kurtosis, skew
variables = ['cs', 'ba_c', 'turnover']

# Group by investment grade and compute stats
def summarize(group):
    summary = {}
    for var in variables:
        vals = group[var].dropna()
        summary[f'{var}_mean'] = vals.mean()
        summary[f'{var}_std'] = vals.std()
        summary[f'{var}_p25'] = vals.quantile(0.25)
        summary[f'{var}_p50'] = vals.median()
        summary[f'{var}_p75'] = vals.quantile(0.75)
        summary[f'{var}_kurtosis'] = kurtosis(vals)
        summary[f'{var}_skewness'] = skew(vals)
    return pd.Series(summary)

summary_df = df.groupby('investment_grade').apply(summarize)
summary_df_transposed = summary_df.T

latex_table = summary_df_transposed.round(3).to_latex(index=True, caption='Summary statistics by investment grade', label='tab:summary_stats_transposed')

with open("summary_stats_table_transposed.tex", "w") as f:
    f.write(latex_table)


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = df.dropna(subset='ba_c')
df['log_turnover_neg'] = -np.log(df['turnover'])
df['neg_ba'] = -df['ba_c']
scaler = StandardScaler()
X = scaler.fit_transform(df[['turnover', 'ba_c']])
X[:, 1] *= -1  # Invert spread axis before PCA (same logic as above)

pca = PCA(n_components=1)
df['liqq'] = pca.fit_transform(X)


pca = PCA(n_components=2)
pcs = pca.fit_transform(X)

df['log_liq_w'] = pca.explained_variance_ratio_[0] * pcs[:, 0] +  pca.explained_variance_ratio_[1] * pcs[:, 1] 

plt.hist(df['liq_w'], bins = 100)
plt.show()

plt.figure(figsize=(8,6))
scatter = plt.scatter(pcs[:, 0], pcs[:, 1], 
                      #c=df['riskiness'], cmap='viridis', 
                      alpha=0.7, edgecolors='k', s= 10)
plt.xlabel("Turnover")
plt.ylabel("Bid-Ask Spread")
plt.grid(True)
plt.show()
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

df = pd.read_csv('data_hfiv.csv')
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])
df['cusip_id'] = df['cusip_id'].astype(str)
df = df.sort_values(['cusip_id', 'trd_exctn_dt'])
df = df.set_index(['cusip_id', 'trd_exctn_dt'])

#df['Delta_rf'] = df.groupby(level='cusip_id')['rf'].diff()
#df = df.reset_index()
mp_df = pd.read_excel('MPshocksAcosta.xlsx', sheet_name='shocks')
df["turnover_rolling"] = (
    df.groupby("cusip_id")["turnover"]
      .rolling(window=30, min_periods=1)
      .median()
      .shift(1)  # ensures only past values are used
      .reset_index(level=0, drop=True)
)
df["ba_c_rolling"] = (
    df.groupby("cusip_id")["ba_c"]
      .rolling(window=30, min_periods=1)
      .median()
      .shift(1)  # ensures only past values are used
      .reset_index(level=0, drop=True)
)

dfFOMC = df.dropna(subset = ['GSS_target', 'GSS_path','ns'])
dep_vars = ['yield_', 'cs', 'turnover', 'ba_c', 'price', 'volume']
shocks = ['GSS_target', 'GSS_path', 'ns']
indep_vars = ['ffr', 'slope']
liq = 'turnover'

def twoway_demean(df, varnames, id_col, time_col):

    # Ensure varnames is a list
    if isinstance(varnames, str):
        varnames = [varnames]

    # Group means for each column
    mean_id   = df.groupby(id_col)[varnames].transform("mean")
    mean_time = df.groupby(time_col)[varnames].transform("mean")

    # Grand means for each column
    grand_mean = df[varnames].mean()

    # Two-way demean formula
    result = df[varnames] - mean_id - mean_time + grand_mean

    # If input was a single column, return a Series
    if len(varnames) == 1:
        return result[varnames[0]]

    return result

def oneway_demean(df, varnames, id_col):
    # Ensure varnames is a list
    if isinstance(varnames, str):
        varnames = [varnames]

    # Compute group means for each column
    mean_id = df.groupby(id_col)[varnames].transform("mean")

    # Compute grand means for each column
    grand_mean = df[varnames].mean()

    # Return demeaned variables
    result = df[varnames] - mean_id + grand_mean

    # If user passed a single variable, return a Series instead of DataFrame
    if len(varnames) == 1:
        return result[varnames[0]]

    return result

models_list = []
for y_var in dep_vars:
        y_1fe  = oneway_demean(dfFOMC, y_var, "cusip_id")
        x_1fe  = oneway_demean(dfFOMC, indep_vars, "cusip_id")
        z_1fe  = oneway_demean(dfFOMC, shocks, "cusip_id")
        y_2fe = twoway_demean(dfFOMC, y_var, "cusip_id", "trd_exctn_dt")
        x_2fe = twoway_demean(dfFOMC, indep_vars, "cusip_id", "trd_exctn_dt")
        z_2fe = twoway_demean(dfFOMC, shocks, "cusip_id", "trd_exctn_dt")
        ig = pd.get_dummies(dfFOMC['ig_hy'], drop_first=True)
        iv_model = IV2SLS(
            dependent=y_1fe,
            exog=ig,
            endog=x_1fe,
            instruments=z_1fe
                ).fit(cov_type='robust')
        iv_twfe = IV2SLS(
            dependent=y_2fe,
            exog=ig,
            endog=x_2fe,
            instruments=z_2fe
            ).fit(cov_type='robust')
        entry = {
                    "y": y_var
                    }

    # Add all beta_x_var and pval_x_var keys dynamically
        for x_var in indep_vars:
                entry[f"beta_{x_var}"] = iv_model.params[x_var]
                entry[f"pval_{x_var}"] = iv_model.pvalues[x_var]
                entry[f"se_{x_var}"] = iv_model.std_errors[x_var]
                entry[f"tstats_{x_var}"] = iv_model.tstats[x_var]
                entry['Time FE'] = 'No'

        models_list.append(entry)
        entry = {
                    "y": y_var
                    }

    # Add all beta_x_var and pval_x_var keys dynamically
        for x_var in indep_vars:
                entry[f"beta_{x_var}"] = iv_twfe.params[x_var]
                entry[f"pval_{x_var}"] = iv_twfe.pvalues[x_var]
                entry[f"se_{x_var}"] = iv_twfe.std_errors[x_var]
                entry[f"tstats_{x_var}"] = iv_twfe.tstats[x_var]
                entry['Time FE'] = 'Yes'

        models_list.append(entry)

results = pd.DataFrame(models_list)

dep_vars = ['yield_', 'cs', 'price']
liq = ['turnover', 'ba_c']

models_list = []
for y_var in dep_vars:
        for liq_var in liq:
            y_1fe  = oneway_demean(dfFOMC, y_var, "cusip_id")
            x_1fe  = oneway_demean(dfFOMC, indep_vars, "cusip_id")
            z_1fe  = oneway_demean(dfFOMC, shocks, "cusip_id")
            liq_1fe = oneway_demean(dfFOMC, f'{liq_var}_rolling', "cusip_id")
            x_liq_1fe = x_1fe.mul(liq_1fe, axis=0)
            x_liq_1fe.columns = [f'{c} x {liq_var}' for c in x_1fe .columns]
            z_liq_1fe = z_1fe.mul(liq_1fe, axis=0)
            z_liq_1fe.columns = [f'{c} x {liq_var}' for c in z_1fe .columns]
            ig_dummies = pd.get_dummies(dfFOMC['ig_hy'], drop_first=True)
            endog = pd.concat([x_1fe, x_liq_1fe], axis=1)
            ivs = pd.concat([z_1fe, z_liq_1fe], axis=1)
            exog = pd.concat([ig_dummies, liq_1fe], axis=1)
            iv_model = IV2SLS(
                        dependent=y_1fe,
                        exog=exog,
                        endog=endog,
                        instruments=ivs
                        ).fit(cov_type='robust')
            interaction_betas = {
                       f"{c} x {liq_var}": iv_model.params[f"{c} x {liq_var}"]
                       for c in indep_vars
                       }

            interaction_pvals = {
                       f"{c} x {liq_var}": iv_model.pvalues[f"{c} x {liq_var}"]
                       for c in indep_vars
                       }
            interaction_se = {
                   f"{c} x {liq_var}": iv_model.std_errors[f"{c} x {liq_var}"]
                   for c in indep_vars
                   }
            interaction_tstats = {
                   f"{c} x {liq_var}": iv_model.tstats[f"{c} x {liq_var}"]
                   for c in indep_vars
                   }
           # store base coefficient
            entry = {
                   "y": y_var,
                   "interaction_betas": interaction_betas,
                   "interaction_pvals": interaction_pvals,
                   "interaction_se": interaction_se,
                   "interaction_tstats": interaction_tstats
                   }

   # Add all beta_x_var and pval_x_var keys dynamically
            for x_var in indep_vars:
               entry[f"beta_{x_var}"] = iv_model.params[x_var]
               entry[f"pval_{x_var}"] = iv_model.pvalues[x_var]
               entry[f"se_{x_var}"] = iv_model.std_errors[x_var]
               entry[f"tstats_{x_var}"] = iv_model.tstats[x_var]
               entry['liq var'] = liq_var

            models_list.append(entry)

results = pd.json_normalize(models_list)


models_list=[]
for y_var in dep_vars:
        for liq_var in liq:
            y_2fe = twoway_demean(dfFOMC, y_var, "cusip_id", "trd_exctn_dt")
            x_2fe = twoway_demean(dfFOMC, indep_vars, "cusip_id", "trd_exctn_dt")
            z_2fe = twoway_demean(dfFOMC, shocks, "cusip_id", "trd_exctn_dt")
            liq_2fe = twoway_demean(dfFOMC, f'{liq_var}_rolling', "cusip_id", "trd_exctn_dt")
            x_liq_2fe = x_2fe.mul(liq_2fe, axis=0)
            x_liq_2fe.columns = [f'{c} x {liq_var}' for c in x_1fe .columns]
            z_liq_2fe = z_2fe.mul(liq_2fe, axis=0)
            z_liq_2fe.columns = [f'{c} x {liq_var}' for c in z_1fe .columns]
            ig_dummies = pd.get_dummies(dfFOMC['ig_hy'], drop_first=True)
            endog = pd.concat([x_2fe, x_liq_2fe], axis=1)
            ivs = pd.concat([z_2fe, z_liq_2fe], axis=1)
            exog = pd.concat([ig_dummies, liq_2fe], axis=1)
            iv_twfe = IV2SLS(
                        dependent=y_2fe,
                        exog=exog,
                        endog=endog,
                        instruments=ivs
                        ).fit(cov_type='robust')
            interaction_betas = {
                       f"{c} x {liq_var}": iv_twfe.params[f"{c} x {liq_var}"]
                       for c in indep_vars
                       }

            interaction_pvals = {
                       f"{c} x {liq_var}": iv_twfe.pvalues[f"{c} x {liq_var}"]
                       for c in indep_vars
                       }
            interaction_se = {
                   f"{c} x {liq_var}": iv_twfe.std_errors[f"{c} x {liq_var}"]
                   for c in indep_vars
                   }
            interaction_tstats = {
                   f"{c} x {liq_var}": iv_twfe.tstats[f"{c} x {liq_var}"]
                   for c in indep_vars
                   }
           # store base coefficient
            entry = {
                   "y": y_var,
                   "interaction_betas": interaction_betas,
                   "interaction_pvals": interaction_pvals,
                   "interaction_se": interaction_se,
                   "interaction_tstats": interaction_tstats
                   }

   # Add all beta_x_var and pval_x_var keys dynamically
            for x_var in indep_vars:
               entry[f"beta_{x_var}"] = iv_twfe.params[x_var]
               entry[f"pval_{x_var}"] = iv_twfe.pvalues[x_var]
               entry[f"se_{x_var}"] = iv_twfe.std_errors[x_var]
               entry[f"tstats_{x_var}"] = iv_twfe.tstats[x_var]
               entry['liq var'] = liq_var

            models_list.append(entry)

results1 = pd.json_normalize(models_list)

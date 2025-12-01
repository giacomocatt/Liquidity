def add_stars(beta, pval):
    if pval < 0.01:
        return f"{beta:.3f}***"
    elif pval < 0.05:
        return f"{beta:.3f}**"
    elif pval < 0.10:
        return f"{beta:.3f}*"
    else:
        return f"{beta:.3f}"
    

def format_entry(beta, se, pval):
    beta_stars = add_stars(beta, pval)
    return f"{beta_stars}\n({se:.3f})"

table_dict = {x: {} for x in indep_vars}

for _, row in results.iterrows():
    y_var = row["y"]

    # ffr entry
    table_dict["ffr"][y_var] = format_entry(
        row["beta_ffr"], row["se_ffr"], row["pval_ffr"]
    )
    
    # slope entry
    table_dict["slope"][y_var] = format_entry(
        row["beta_slope"], row["se_slope"], row["pval_slope"]
    )
# Convert into DataFrame
latex_df = pd.DataFrame(table_dict).T

latex_table = latex_df.to_latex(
    index=True,
    header=True,
    escape=False,    # KEEP stars
    column_format="lccc"  # adjust depending on number of outcomes
)

covariates = ['ffr', 'slope', 'ffr x turnover', 'ffr x ba', 'slope x turnover', 'slope x ba']
table_dict = {x: {} for x in covariates}

for _, row in results1.iterrows():
    y_var = row["y"]

    # assuming liq_var is a string column in results
    liq = str(row["liq var"])

    table_dict["ffr"][f"{y_var}_{liq}"] = format_entry(
        row["beta_ffr"], row["se_ffr"], row["pval_ffr"]
    )
    table_dict["ffr x turnover"][f"{y_var}_{liq}"] = format_entry(
        row["interaction_betas.ffr x turnover"], 
        row["interaction_se.ffr x turnover"], 
        row["interaction_pvals.ffr x turnover"]
    )
    table_dict["ffr x ba"][f"{y_var}_{liq}"] = format_entry(
        row["interaction_betas.ffr x ba_c"], 
        row["interaction_se.ffr x ba_c"], 
        row["interaction_pvals.ffr x ba_c"]
    )
    
    # slope entry
    table_dict["slope"][f"{y_var}_{liq}"] = format_entry(
        row["beta_slope"], row["se_slope"], row["pval_slope"]
    )
    table_dict["slope x turnover"][f"{y_var}_{liq}"] = format_entry(
        row["interaction_betas.slope x turnover"], 
        row["interaction_se.slope x turnover"], 
        row["interaction_pvals.slope x turnover"]
    )
    table_dict["slope x ba"][f"{y_var}_{liq}"] = format_entry(
        row["interaction_betas.slope x ba_c"], 
        row["interaction_se.slope x ba_c"], 
        row["interaction_pvals.slope x ba_c"]
    )
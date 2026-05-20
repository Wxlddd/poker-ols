import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools

def run_best_subset_selection(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("BEST SUBSET SELECTION (OLS SUBMODEL OPTIMIZATION)")
    print("=" * 60)
    
    variables = list(X_train.columns)
    results = []
    
    # Try all non-empty subsets of variables
    for k in range(1, len(variables) + 1):
        for subset in itertools.combinations(variables, k):
            subset = list(subset)
            # Fit model on training set
            X_train_sub = sm.add_constant(X_train[subset])
            model_sub = sm.OLS(y_train, X_train_sub).fit()
            
            # Predict on test set
            X_test_sub = sm.add_constant(X_test[subset])
            y_pred_sub = model_sub.predict(X_test_sub)
            
            mse_sub = mean_squared_error(y_test, y_pred_sub)
            r2_test_sub = r2_score(y_test, y_pred_sub)
            
            results.append({
                'subset': subset,
                'k': k,
                'r2': model_sub.rsquared,
                'adj_r2': model_sub.rsquared_adj,
                'aic': model_sub.aic,
                'bic': model_sub.bic,
                'test_mse': mse_sub,
                'test_r2': r2_test_sub,
                'model': model_sub
            })
            
    # Rank by Adjusted R2
    results_sorted = sorted(results, key=lambda x: x['adj_r2'], reverse=True)
    
    print(f"Tested all {len(results)} non-empty variable combinations.")
    print("\n--- Top 5 Submodels Ranked by Adjusted R-squared ---")
    for idx, res in enumerate(results_sorted[:5]):
        vars_str = ", ".join(res['subset'])
        print(f"Rank {idx+1}: Adjusted R2 = {res['adj_r2']:.4f} (R2 = {res['r2']:.4f}, k={res['k']})")
        print(f"        AIC = {res['aic']:.2f}, BIC = {res['bic']:.2f}")
        print(f"        Test Set MSE = {res['test_mse']:.2f}, Test Set R2 = {res['test_r2']:.4f}")
        print(f"        Variables: [{vars_str}]")
        print("-" * 50)
        
    best_model_res = results_sorted[0]
    
    print("\n--- Optimal Submodel Summary ---")
    print(best_model_res['model'].summary())
    return best_model_res

def run_analysis():
    print("=" * 60)
    print("POKER PLAYER STATISTICAL ANALYSIS PIPELINE (OLS)")
    print("=" * 60)
    
    # 1. Data Cleaning & Filtering
    print("\n[STEP 1] Data Loading and Cleaning...")
    csv_file = 'poker_stats.csv'
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset file '{csv_file}' not found. Please run get_dataset.py first.")
        
    df = pd.read_csv(csv_file)
    print(f"Original dataset shape: {df.shape}")
    
    # Filter players with significant hand sample size (Hands > 200)
    # to reduce win rate variance and focus on regulars / high-volume players
    min_hands = 200
    df_filtered = df[df['Hands'] > min_hands].copy()
    print(f"Dataset shape after filtering (Hands > {min_hands}): {df_filtered.shape}")
    
    # Handle NaNs if any
    initial_len = len(df_filtered)
    df_filtered = df_filtered.dropna()
    print(f"Dataset shape after dropping NaNs: {df_filtered.shape}")
    if len(df_filtered) < initial_len:
        print(f"Dropped {initial_len - len(df_filtered)} rows containing null values.")
        
    # Variables definition
    X_cols = ['VPIP', 'PFR', '3Bet', 'Postflop_Agg', 'WTSD', 'W$SD', 'Hands']
    y_col = 'WinRate'
    
    # 2. Train/Test Split
    print("\n[STEP 2] Splitting into Training (80%) and Test (20%) Sets...")
    X = df_filtered[X_cols]
    y = df_filtered[y_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} players")
    print(f"Test set size: {X_test.shape[0]} players")
    
    # 3. Diagnostica della Multicollinearità (VIF)
    print("\n[STEP 3] Multicollinearity Diagnostics on Training Set...")
    # Correlation Matrix
    corr_matrix = X_train.corr()
    print("\n--- Correlation Matrix ---")
    print(corr_matrix.round(3))
    
    # VIF calculation
    # statsmodels requires an intercept to compute VIF accurately for the variables
    X_train_vif = sm.add_constant(X_train)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_train_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
    
    print("\n--- Variance Inflation Factor (VIF) ---")
    print(vif_data.to_string(index=False))
    
    # Highlight potential multicollinearity
    high_vif = vif_data[vif_data['VIF'] > 5.0]
    if not high_vif.empty:
        print("\nWARNING: High multicollinearity detected (VIF > 5) for the following variables:")
        print(high_vif.to_string(index=False))
    else:
        print("\nNo critical multicollinearity detected (VIF < 5 for all independent variables).")
        
    # 4. OLS Model Fitting
    print("\n[STEP 4] Fitting OLS Regression Model on Training Set...")
    # Add constant term (intercept)
    X_train_ols = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_ols).fit()
    
    print("\n--- Model Estimation Summary ---")
    print(model.summary())
    
    # 5. Evaluation and Advanced Diagnostics
    print("\n[STEP 5] Model Evaluation & Information Criteria...")
    X_test_ols = sm.add_constant(X_test)
    y_pred = model.predict(X_test_ols)
    
    mse = mean_squared_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    
    print(f"Test Set Mean Squared Error (MSE): {mse:.4f}")
    print(f"Test Set R-squared (R2): {r2_test:.4f}")
    print(f"Akaike Information Criterion (AIC): {model.aic:.2f}")
    print(f"Bayesian Information Criterion (BIC): {model.bic:.2f}")
    
    print("\n[STEP 6] Advanced Regression Diagnostics (Outliers, Leverage & Influence)...")
    
    # Calculate influence, leverage and Cook's distance
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d, _ = influence.cooks_distance
    student_resid = influence.resid_studentized_external
    
    # Sample and parameter count
    n = X_train_ols.shape[0]
    p = X_train_ols.shape[1] # includes constant
    
    # Thresholds
    leverage_thresh = 2 * p / n
    cooks_thresh = 4 / n
    outlier_thresh = 2.0
    
    print(f"Sample Size (n): {n}")
    print(f"Number of parameters (p): {p}")
    print(f"Leverage Threshold (2p/n): {leverage_thresh:.4f}")
    print(f"Cook's Distance Threshold (4/n): {cooks_thresh:.4f}")
    
    # Create a DataFrame of diagnostics for training set
    diag_df = pd.DataFrame({
        'Player': df_filtered.loc[X_train.index, 'Player'],
        'WinRate': y_train,
        'Fitted': model.fittedvalues,
        'Residual': model.resid,
        'Studentized_Resid': student_resid,
        'Leverage': leverage,
        'Cooks_D': cooks_d
    })
    
    # Identify outliers, leverage points, and influential points
    outliers = diag_df[np.abs(diag_df['Studentized_Resid']) > outlier_thresh]
    high_leverage = diag_df[diag_df['Leverage'] > leverage_thresh]
    high_influence = diag_df[diag_df['Cooks_D'] > cooks_thresh]
    
    print(f"\n--- Outlier Diagnostics (Studentized Residual |r| > {outlier_thresh}) ---")
    print(f"Total Outliers: {len(outliers)} ({len(outliers)/n*100:.1f}% of training sample)")
    if len(outliers) > 0:
        print(outliers[['Player', 'WinRate', 'Studentized_Resid']].sort_values(by='Studentized_Resid', key=abs, ascending=False).head(5).to_string(index=False))
        
    print(f"\n--- Leverage Diagnostics (Hat Value > {leverage_thresh:.4f}) ---")
    print(f"Total High Leverage Points: {len(high_leverage)} ({len(high_leverage)/n*100:.1f}% of training sample)")
    if len(high_leverage) > 0:
        print(high_leverage[['Player', 'WinRate', 'Leverage']].sort_values(by='Leverage', ascending=False).head(5).to_string(index=False))
        
    print(f"\n--- Influence Diagnostics (Cook's Distance > {cooks_thresh:.4f}) ---")
    print(f"Total Highly Influential Points: {len(high_influence)} ({len(high_influence)/n*100:.1f}% of training sample)")
    if len(high_influence) > 0:
        print(high_influence[['Player', 'WinRate', 'Cooks_D']].sort_values(by='Cooks_D', ascending=False).head(5).to_string(index=False))
        
    # Generate Advanced 6-Panel Diagnostic Plots
    print("\nGenerating Advanced Diagnostic Plots (6-panel grid)...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(diag_df['Fitted'], diag_df['Residual'], alpha=0.6, color='royalblue', edgecolor='none')
    axes[0, 0].axhline(0, color='red', linestyle='--', lw=1.5)
    axes[0, 0].set_title('1. Residuals vs Fitted')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Raw Residuals')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. Normal Q-Q Plot of Studentized Residuals
    stats.probplot(diag_df['Studentized_Resid'], dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_color('royalblue')
    axes[0, 1].get_lines()[0].set_alpha(0.6)
    axes[0, 1].set_title('2. Normal Q-Q Plot (Studentized Residuals)')
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    # 3. Histogram of Residuals with KDE & Normal Distribution
    axes[1, 0].hist(diag_df['Studentized_Resid'], bins=20, density=True, alpha=0.6, color='royalblue', edgecolor='white')
    # Fit normal distribution
    mu, std = stats.norm.fit(diag_df['Studentized_Resid'])
    xmin, xmax = axes[1, 0].get_xlim()
    x_range = np.linspace(xmin, xmax, 100)
    p_norm = stats.norm.pdf(x_range, mu, std)
    axes[1, 0].plot(x_range, p_norm, 'r-', lw=2, label=f'Normal Fit\n(μ={mu:.2f}, σ={std:.2f})')
    # KDE
    diag_df['Studentized_Resid'].plot(kind='density', ax=axes[1, 0], color='darkorange', lw=2, label='KDE')
    axes[1, 0].set_title('3. Studentized Residuals Distribution')
    axes[1, 0].set_xlabel('Studentized Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # 4. Cook's Distance Plot
    axes[1, 1].stem(diag_df.index, diag_df['Cooks_D'], markerfmt=' ', basefmt=" ")
    axes[1, 1].axhline(cooks_thresh, color='red', linestyle='--', lw=1.5, label=f'Threshold (4/n = {cooks_thresh:.4f})')
    axes[1, 1].set_title("4. Cook's Distance per Observation")
    axes[1, 1].set_xlabel('Observation Index')
    axes[1, 1].set_ylabel("Cook's D")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    # 5. Leverage (Hat values) vs Studentized Residuals
    axes[2, 0].scatter(diag_df['Leverage'], diag_df['Studentized_Resid'], alpha=0.6, color='royalblue', edgecolor='none')
    axes[2, 0].axhline(0, color='black', linestyle='-', lw=1)
    axes[2, 0].axhline(outlier_thresh, color='red', linestyle=':', lw=1.5)
    axes[2, 0].axhline(-outlier_thresh, color='red', linestyle=':', lw=1.5)
    axes[2, 0].axvline(leverage_thresh, color='red', linestyle='--', lw=1.5, label=f'Leverage Thresh ({leverage_thresh:.4f})')
    axes[2, 0].set_title('5. Studentized Residuals vs Leverage')
    axes[2, 0].set_xlabel('Leverage (Hat Values)')
    axes[2, 0].set_ylabel('Studentized Residuals')
    # Label top 3 Cook's D points in the plot
    top_cooks = diag_df.nlargest(3, 'Cooks_D')
    for idx, row in top_cooks.iterrows():
        axes[2, 0].annotate(row['Player'][:6] + '...', (row['Leverage'], row['Studentized_Resid']), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    axes[2, 0].legend()
    axes[2, 0].grid(True, linestyle=':', alpha=0.6)
    
    # 6. Cook's D vs Leverage
    axes[2, 1].scatter(diag_df['Leverage'], diag_df['Cooks_D'], alpha=0.6, color='royalblue', edgecolor='none')
    axes[2, 1].axvline(leverage_thresh, color='red', linestyle='--', lw=1.5)
    axes[2, 1].axhline(cooks_thresh, color='red', linestyle='--', lw=1.5, label=f'Cooks Thresh ({cooks_thresh:.4f})')
    axes[2, 1].set_title("6. Cook's Distance vs Leverage")
    axes[2, 1].set_xlabel('Leverage (Hat Values)')
    axes[2, 1].set_ylabel("Cook's D")
    axes[2, 1].legend()
    axes[2, 1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plots_path = 'residual_diagnostics.png'
    plt.savefig(plots_path, dpi=150)
    plt.close()
    print(f"Advanced diagnostic plots saved to '{plots_path}'.")
    
    # Breusch-Pagan Test for Heteroscedasticity
    print("\nPerforming Breusch-Pagan Test...")
    bp_test = het_breuschpagan(diag_df['Residual'], X_train_ols)
    labels = ['Lagrange Multiplier statistic', 'p-value', 'f-value', 'f p-value']
    bp_results = dict(zip(labels, bp_test))
    
    print("\n--- Breusch-Pagan Test Results ---")
    for key, val in bp_results.items():
        print(f"{key}: {val:.6f}")
        
    if bp_results['p-value'] < 0.05:
        print("\nConclusion: Reject the null hypothesis of homoscedasticity at the 5% significance level.")
        print("There is evidence of HETEROSCEDASTICITY in the residuals.")
        print("Poker win rates naturally have heteroscedasticity because the variance of the observed win rate")
        print("is inversely proportional to the number of hands played (Hands).")
    else:
        print("\nConclusion: Fail to reject the null hypothesis of homoscedasticity.")
        print("The residuals are homoscedastic.")
    # 6. Best Subset Selection (OLS Submodel Optimization)
    best_submodel = run_best_subset_selection(X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == '__main__':
    run_analysis()

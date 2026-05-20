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
    
    # 5. Evaluation and Residual Diagnostics
    print("\n[STEP 5] Model Evaluation on Test Set...")
    X_test_ols = sm.add_constant(X_test)
    y_pred = model.predict(X_test_ols)
    
    mse = mean_squared_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    
    print(f"Test Set Mean Squared Error (MSE): {mse:.4f}")
    print(f"Test Set R-squared (R2): {r2_test:.4f}")
    
    # Diagnostic Plots
    print("\nGenerating Diagnostic Plots...")
    residuals = model.resid
    fitted_vals = model.fittedvalues
    
    # Residuals vs Fitted Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(fitted_vals, residuals, alpha=0.5, color='royalblue', edgecolor='none')
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Normal Q-Q Plot
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normal Q-Q Plot')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plots_path = 'residual_diagnostics.png'
    plt.savefig(plots_path, dpi=150)
    plt.close()
    print(f"Diagnostic plots saved to '{plots_path}'.")
    
    # Breusch-Pagan Test for Heteroscedasticity
    print("\nPerforming Breusch-Pagan Test...")
    bp_test = het_breuschpagan(residuals, X_train_ols)
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
        
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == '__main__':
    run_analysis()

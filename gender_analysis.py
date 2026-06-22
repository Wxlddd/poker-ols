import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.special import inv_boxcox
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score

# Set plot aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150
})

# Path for artifacts
ARTIFACT_DIR = r"C:\Users\loren\.gemini\antigravity\brain\2df59a04-f39e-41fb-b884-2a57332edee4"

def copy_to_artifacts(filename):
    if os.path.exists(filename) and os.path.exists(ARTIFACT_DIR):
        shutil.copy(filename, os.path.join(ARTIFACT_DIR, os.path.basename(filename)))
        print(f"Copied {filename} to artifacts directory.")

def main():
    print("=" * 70)
    print("      END-TO-END STATISTICS WORKFLOW: GENDER PAY GAP ANALYSIS")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: DATA PREPARATION & FEATURE ENGINEERING (Gender Classification)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Data Preparation & Cleaning ---")
    os.makedirs('plots', exist_ok=True)
    if not os.path.exists("Salaries.csv"):
        raise FileNotFoundError("Salaries.csv not found in the workspace!")
        
    df = pd.read_csv("Salaries.csv", low_memory=False)
    print(f"Original dataset shape: {df.shape}")
    
    # Convert pay columns to numeric
    df['BasePay'] = pd.to_numeric(df['BasePay'], errors='coerce')
    df['TotalPay'] = pd.to_numeric(df['TotalPay'], errors='coerce')
    
    # Filter for strictly positive salaries as requested
    clean_df = df[(df['BasePay'] > 0) & (df['TotalPay'] > 0)].copy()
    print(f"Dataset shape after filtering positive pay: {clean_df.shape}")
    
    # Feature Engineering: Seniority (Anzianità)
    # Standardize names to lower case and strip whitespace to link records across years
    clean_df['CleanName'] = clean_df['EmployeeName'].astype(str).str.lower().str.strip()
    first_year = clean_df.groupby('CleanName')['Year'].transform('min')
    clean_df['Seniority'] = clean_df['Year'] - first_year
    print("\nSeniority (tenure proxy) distribution:")
    print(clean_df['Seniority'].value_counts())
    
    # Feature Engineering: Job Macro-Categories
    def categorize_job(title):
        t = str(title).lower()
        if any(k in t for k in ['police', 'sheriff', 'sergeant', 'officer', 'patrol', 'jailer', 'lieutenant', 'captain']):
            return 'Police'
        elif any(k in t for k in ['fire', 'firefighter', 'paramedic', 'emt']):
            return 'Fire'
        elif any(k in t for k in ['nurse', 'medical', 'physician', 'health', 'patient care', 'clinical', 'therapist', 'dentist', 'psychiatrist', 'pharmacist']):
            return 'Medical'
        elif any(k in t for k in ['clerk', 'admin', 'secretary', 'analyst', 'assistant', 'coordinator', 'office', 'accountant', 'auditor', 'manager', 'attorney', 'lawyer', 'counsel', 'registrar', 'cashier']):
            return 'Admin'
        elif any(k in t for k in ['teacher', 'professor', 'educator', 'library', 'recreation', 'aide', 'student', 'trainee', 'coach', 'instructor', 'school']):
            return 'Education'
        else:
            return 'Other'
            
    clean_df['JobCategory'] = clean_df['JobTitle'].apply(categorize_job)
    print("\nJob Macro-Categories distribution:")
    print(clean_df['JobCategory'].value_counts())
    
    # Feature Engineering: Gender Classification (Optimized Unique Name Mapping)
    print("\nClassifying Gender based on first names...")
    import gender_guesser.detector as gd
    detector = gd.Detector(case_sensitive=True)
    
    # Clean first names and extract
    def extract_first_name(full_name):
        name_parts = str(full_name).split()
        if len(name_parts) == 0:
            return ""
        # Extract first word, remove punctuation
        first_word = re.sub(r'[^a-zA-Z]', '', name_parts[0])
        return first_word.title()
        
    clean_df['FirstName'] = clean_df['EmployeeName'].apply(extract_first_name)
    
    # Unique names set optimization
    unique_names = clean_df['FirstName'].unique()
    print(f"Found {len(unique_names)} unique first names. Resolving genders...")
    
    gender_map = {}
    for name in unique_names:
        if len(name) <= 2:  # Exclude single character initials (A., J., etc.)
            gender_map[name] = np.nan
            continue
        g = detector.get_gender(name)
        if g in ['male', 'mostly_male']:
            gender_map[name] = 0.0
        elif g in ['female', 'mostly_female']:
            gender_map[name] = 1.0
        else:
            gender_map[name] = np.nan # drop 'unknown' and 'andy' (ambiguous)
            
    clean_df['Gender'] = clean_df['FirstName'].map(gender_map)
    
    # Drop ambiguous/unknown gender records
    final_df = clean_df.dropna(subset=['Gender']).copy()
    final_df['Gender'] = final_df['Gender'].astype(int)
    print(f"\nFinal dataset shape after dropping ambiguous genders: {final_df.shape}")
    print("Gender breakdown (1 = Female, 0 = Male):")
    print(final_df['Gender'].value_counts(normalize=True) * 100)
    
    # Generate Gender Pay Gap by Job Category Plot
    print("\nGenerating Gender Pay Gap by Job Category Plot...")
    plt.figure(figsize=(12, 6))
    plot_df = final_df.copy()
    plot_df['Genere'] = plot_df['Gender'].map({0: 'Uomo (0)', 1: 'Donna (1)'})
    sns.boxplot(
        data=plot_df, 
        x='JobCategory', 
        y='TotalPay', 
        hue='Genere', 
        palette={'Uomo (0)': '#1f77b4', 'Donna (1)': '#e377c2'},
        showfliers=False
    )
    plt.title('Distribuzione della Retribuzione Totale per Macro-Categoria e Genere')
    plt.xlabel('Macro-Categoria Lavorativa')
    plt.ylabel('Total Pay ($)')
    plt.legend(title='Genere')
    plt.tight_layout()
    plt.savefig('plots/gender_pay_gap_by_job.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/gender_pay_gap_by_job.png')

    # Generate Seniority Pay Gap Plot
    print("Generating Seniority Pay Gap Plot...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df, 
        x='Seniority', 
        y='TotalPay', 
        hue='Genere', 
        marker='o',
        linewidth=2.5,
        palette={'Uomo (0)': '#1f77b4', 'Donna (1)': '#e377c2'},
        errorbar=('ci', 95)
    )
    plt.title('Andamento Medio della Retribuzione Totale per Anzianità e Genere')
    plt.xlabel('Anzianità (Proxy Anni di Servizio)')
    plt.ylabel('Retribuzione Totale Media ($)')
    plt.xticks(sorted(plot_df['Seniority'].unique()))
    plt.legend(title='Genere')
    plt.tight_layout()
    plt.savefig('plots/seniority_pay_gap.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/seniority_pay_gap.png')
    
    # -------------------------------------------------------------------------
    # STEP 2: DESIGN MATRIX (Z) SPECIFICATION & RANK VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Design Matrix (Z) Construction & Rank Verification ---")
    Y = final_df['TotalPay'].values
    
    # Construct base columns
    design_data = pd.DataFrame()
    design_data['Intercept'] = np.ones(len(final_df))
    design_data['Gender'] = final_df['Gender'].values
    design_data['Seniority'] = final_df['Seniority'].values
    
    # Year Dummies (Reference: 2011)
    year_dummies = pd.get_dummies(final_df['Year'], prefix='Year', drop_first=True, dtype=int)
    for col in year_dummies.columns:
        design_data[col] = year_dummies[col].values
        
    # Job Category Dummies (Reference: Other)
    job_dummies = pd.get_dummies(final_df['JobCategory'], prefix='Job', drop_first=True, dtype=int)
    # Ensure standard order and drop Job_Other explicitly if present
    if 'Job_Other' in job_dummies.columns:
        job_dummies = job_dummies.drop(columns=['Job_Other'])
    for col in job_dummies.columns:
        design_data[col] = job_dummies[col].values
        
    # Interaction terms
    # 1. Gender * Seniority
    design_data['Gender_x_Seniority'] = design_data['Gender'] * design_data['Seniority']
    # 2. Gender * JobCategory
    for col in job_dummies.columns:
        inter_name = f"Gender_x_{col}"
        design_data[inter_name] = design_data['Gender'] * design_data[col]
        
    Z = design_data.copy()
    print("Design Matrix Z Columns:")
    print(Z.columns.tolist())
    print(f"Design Matrix dimensions: {Z.shape}")
    
    # Rank Verification via SVD
    U, s, Vt = np.linalg.svd(Z.values, full_matrices=False)
    rank = np.sum(s > 1e-10)
    num_cols = Z.shape[1]
    print(f"Matrix Rank: {rank} (out of {num_cols} columns)")
    if rank == num_cols:
        print("Success: Z has full column rank. Z^T Z is non-singular and invertible.")
    else:
        print("Warning: Z is rank deficient! Multicollinearity exists.")
        
    # -------------------------------------------------------------------------
    # STEP 3: OLS ESTIMATION & VARIANCE DECOMPOSITION
    # -------------------------------------------------------------------------
    print("\n--- Step 3: OLS Estimation & Variance Decomposition ---")
    model_naive = sm.OLS(Y, Z)
    results_naive = model_naive.fit()
    print(results_naive.summary())
    
    # Variance Decomposition: SSTOT = SSREG + SSRES
    y_hat = results_naive.fittedvalues
    residuals = Y - y_hat
    y_bar = np.mean(Y)
    
    SSTOT = np.sum((Y - y_bar) ** 2)
    SSREG = np.sum((y_hat - y_bar) ** 2)
    SSRES = np.sum(residuals ** 2)
    
    print("\nVariance Decomposition Checks:")
    print(f"SSTOT: {SSTOT:,.2f}")
    print(f"SSREG: {SSREG:,.2f}")
    print(f"SSRES: {SSRES:,.2f}")
    print(f"SSREG + SSRES: {(SSREG + SSRES):,.2f}")
    print(f"Difference SSTOT - (SSREG + SSRES): {SSTOT - (SSREG + SSRES):.8e}")
    
    # Check orthogonality of fitted values and residuals: y_hat^T * e = 0
    ortho = np.dot(y_hat, residuals)
    print(f"Orthogonality y_hat^T * e: {ortho:.8e} (should be approx 0)")
    
    # -------------------------------------------------------------------------
    # STEP 4: DIAGNOSTICA AVANZATA (Gauss-Markov Verification)
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Advanced Diagnostics & Gauss-Markov Hypothesis ---")
    
    # Vectorized leverage calculation: h_ii = diag(Z(Z^TZ)^-1 Z^T)
    # Avoids n x n matrix construction to conserve memory (O(n) space instead of O(n^2))
    print("Computing Hat Matrix diagonals (leverage)...")
    Z_arr = Z.values
    inv_ZTZ = np.linalg.inv(Z_arr.T @ Z_arr)
    h = np.sum((Z_arr @ inv_ZTZ) * Z_arr, axis=1)
    
    n, r_plus_1 = Z.shape
    leverage_threshold = 2 * r_plus_1 / n
    critical_leverage_count = np.sum(h > leverage_threshold)
    print(f"Leverage threshold 2(r+1)/n: {leverage_threshold:.6f}")
    print(f"Number of critical leverage points: {critical_leverage_count} ({critical_leverage_count / n * 100:.2f}%)")
    
    # Studentized Residuals
    print("Computing Studentized Residuals and Cook's Distance...")
    sigma_hat = np.sqrt(SSRES / (n - r_plus_1))
    studentized_residuals = residuals / (sigma_hat * np.sqrt(1.0 - h))
    
    # Cook's Distance: D_i = (t_i^2 / (r+1)) * (h_ii / (1 - h_ii))
    cooks_d = (studentized_residuals ** 2 / r_plus_1) * (h / (1.0 - h))
    
    # Identify top 5 outliers by Cook's distance
    top_5_idx = np.argsort(cooks_d)[-5:][::-1]
    top_5_outliers = final_df.iloc[top_5_idx].copy()
    top_5_outliers['Cooks_D'] = cooks_d[top_5_idx]
    top_5_outliers['Student_Res'] = studentized_residuals[top_5_idx]
    print("\nTop 5 Influential Outliers by Cook's Distance:")
    print(top_5_outliers[['EmployeeName', 'JobTitle', 'TotalPay', 'Gender', 'Cooks_D', 'Student_Res']])
    
    # Generate Advanced Leverage and Cook's Distance Plot
    print("\nGenerating Leverage and Cook's Distance plots...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Leverage Index Plot
    axes[0].scatter(
        range(n), 
        h, 
        color='#1f77b4', 
        s=2, 
        alpha=0.2, 
        label='Leverage ($h_{ii}$)',
        rasterized=True
    )
    axes[0].axhline(leverage_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Soglia: {leverage_threshold:.6f}')
    axes[0].set_title("Index Plot dei Punti di Leva (Leverage)")
    axes[0].set_xlabel("Indice Osservazione")
    axes[0].set_ylabel("Valore di Leva ($h_{ii}$)")
    axes[0].legend()
    
    # Plot 2: Cook's Distance Index Plot
    axes[1].scatter(
        range(n), 
        cooks_d, 
        color='#d62728', 
        s=2, 
        alpha=0.2, 
        label="Distanza di Cook ($D_i$)",
        rasterized=True
    )
    axes[1].set_title("Index Plot della Distanza di Cook ($D_i$)")
    axes[1].set_xlabel("Indice Osservazione")
    axes[1].set_ylabel("Distanza di Cook ($D_i$)")
    
    # Set Y-limit to make room for staggered labels
    axes[1].set_ylim(0, max(cooks_d) * 1.8)
    
    # Annotate top 5 outliers with staggered positions to prevent horizontal/vertical overlapping
    for i, idx in enumerate(top_5_idx):
        name = final_df.iloc[idx]['EmployeeName']
        val = cooks_d[idx]
        # Stagger horizontally and vertically based on order
        h_offset = -15000 if i % 2 == 0 else 15000
        v_offset = 0.0003 + (i * 0.0003)
        axes[1].annotate(
            name, 
            xy=(idx, val), 
            xytext=(idx + h_offset, val + v_offset),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7)
        )
    axes[1].legend()
    
    plt.suptitle("Analisi Statistica di Diagnostica Avanzata: Leverage e Influenza (Distanza di Cook)", fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/leverage_cooks.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/leverage_cooks.png')
    
    # Heteroscedasticity: Breusch-Pagan Test
    print("\nRunning Breusch-Pagan Test...")
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = het_breuschpagan(residuals, Z)
    print(f"Breusch-Pagan LM Stat: {bp_stat:.4f}")
    print(f"Breusch-Pagan p-value: {bp_pvalue:.8e}")
    if bp_pvalue < 0.05:
        print("Result: Reject H0 (Homoscedasticity). Strong evidence of eteroschedasticity!")
    else:
        print("Result: Fail to reject H0. Residuals are homoscedastic.")
        
    # Normality: Shapiro-Wilk Test (on a subsample of 5000 residuals due to large N sensitivity and limits)
    print("\nRunning Shapiro-Wilk Normality Test on a random subsample of 5000 residuals (Naïve Model)...")
    np.random.seed(42)
    subsample_res = np.random.choice(residuals, 5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(subsample_res)
    print(f"Shapiro-Wilk W-statistic: {shapiro_stat:.6f}")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.8e}")
    if shapiro_p < 0.05:
        print("Result: Reject H0 (Normality). Residuals are NOT normally distributed!")
    else:
        print("Result: Fail to reject H0 (Normality). Residuals are approximately normal.")
        
    # Generate OLS Diagnostic Plots
    print("\nGenerating OLS Diagnostic Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subsample 10,000 random points for visualization to avoid dense cluttering
    sample_indices = np.random.choice(n, size=min(10000, n), replace=False)
    
    # Plot 1: Residuals vs Fitted
    sns.scatterplot(
        x=y_hat[sample_indices],
        y=residuals[sample_indices],
        ax=axes[0],
        alpha=0.25,
        edgecolor='none',
        color='#1f77b4'
    )
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Residuals vs Fitted (Subsampled 10k)')
    axes[0].set_xlabel('Fitted Values ($y_{hat}$)')
    axes[0].set_ylabel('Residuals ($e$)')
    
    # Plot 2: Normal Q-Q Plot
    # Standardize studentized residuals and compare quantiles
    sorted_stud_res = np.sort(studentized_residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.5 / n, 1.0 - 0.5 / n, n))
    
    sns.scatterplot(
        x=theoretical_quantiles[sample_indices],
        y=sorted_stud_res[sample_indices],
        ax=axes[1],
        alpha=0.25,
        edgecolor='none',
        color='#2ca02c'
    )
    # Fit line through 25th and 75th percentiles
    x25, x75 = np.percentile(theoretical_quantiles, [25, 75])
    y25, y75 = np.percentile(sorted_stud_res, [25, 75])
    slope = (y75 - y25) / (x75 - x25)
    intercept = y25 - slope * x25
    axes[1].plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, color='red', linestyle='-', linewidth=1.5)
    
    axes[1].set_title('Normal Q-Q Plot')
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Studentized Residuals')
    
    plt.tight_layout()
    plt.savefig('plots/ols_diagnostics.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/ols_diagnostics.png')
    
    # -------------------------------------------------------------------------
    # STEP 5: BOX-COX TRANSFORMATION FOR HETEROSCEDASTICITY
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Box-Cox Transformation for Heteroscedasticity ---")
    
    # Find optimal lambda using MLE
    opt_lambda = stats.boxcox_normmax(Y, method='mle')
    print(f"Optimal Lambda (MLE): {opt_lambda:.4f}")
    
    # Let's plot the profile log-likelihood for lambda as requested
    lambdas = np.linspace(-1.5, 1.5, 100)
    log_likelihoods = []
    # Log likelihood of Box-Cox: L(lambda) = -n/2 * ln(var(y_lambda)) + (lambda - 1) * sum(ln(y))
    sum_ln_y = np.sum(np.log(Y))
    for lmbda in lambdas:
        if lmbda == 0:
            y_trans = np.log(Y)
        else:
            y_trans = (Y**lmbda - 1) / lmbda
        var_trans = np.var(y_trans)
        ll = -n/2 * np.log(var_trans) + (lmbda - 1) * sum_ln_y
        log_likelihoods.append(ll)
        
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, log_likelihoods, color='purple', linewidth=2, label='Profile Log-Likelihood')
    plt.axvline(opt_lambda, color='red', linestyle='--', label=f'Optimal $\lambda$ = {opt_lambda:.4f}')
    plt.axvline(0, color='darkorange', linestyle=':', label='$\lambda$ = 0 (Log Trans)')
    plt.title('Box-Cox Profile Log-Likelihood')
    plt.xlabel('$\lambda$ Parameter')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/boxcox_likelihood.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/boxcox_likelihood.png')
    
    # Econometric custom transformation decision
    # Round optimal lambda to the nearest highly interpretable standard value (0.5 for square root)
    print(f"Optimal Lambda (MLE) is {opt_lambda:.4f}.")
    print("For econometric interpretability, we round it to the nearest standard value: lambda = 0.5 (Square Root Transformation).")
    opt_lambda = 0.5
    Y_trans = stats.boxcox(Y, lmbda=0.5)
        
    # Generate Salary Distribution Plot (Raw vs Transformed)
    print("\nGenerating Salary Distribution plot (raw vs transformed)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(Y, kde=True, ax=axes[0], color='#1f77b4', bins=50)
    axes[0].set_title('Distribuzione Salari Originali (Raw TotalPay)')
    axes[0].set_xlabel('Total Pay ($)')
    axes[0].set_ylabel('Frequenza')
    
    sns.histplot(Y_trans, kde=True, ax=axes[1], color='#9467bd', bins=50)
    axes[1].set_title(f'Distribuzione Salari Trasformati ($\lambda$ = {opt_lambda:.4f})')
    axes[1].set_xlabel('Total Pay Trasformato')
    axes[1].set_ylabel('Frequenza')
    
    plt.suptitle('Impatto della Trasformazione Box-Cox sulla Distribuzione Salariale', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/salary_distribution.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/salary_distribution.png')
        
    # Refit OLS with transformed Y
    model_trans = sm.OLS(Y_trans, Z)
    results_trans = model_trans.fit()
    print("\n--- OLS SUMMARY ON TRANSFORMED VARIABLE ---")
    print(results_trans.summary())
    
    y_hat_trans = results_trans.fittedvalues
    residuals_trans = Y_trans - y_hat_trans
    
    # Breusch-Pagan test on transformed model
    bp_stat_t, bp_pvalue_t, bp_fstat_t, bp_fpvalue_t = het_breuschpagan(residuals_trans, Z)
    print("\nBreusch-Pagan Test on Transformed Model:")
    print(f"Breusch-Pagan LM Stat: {bp_stat_t:.4f}")
    print(f"Breusch-Pagan p-value: {bp_pvalue_t:.8e}")
    if bp_pvalue_t < 0.05:
        print("Result: Reject H0. Some heteroscedasticity remains (common with massive datasets), but check plot improvements.")
    else:
        print("Result: Fail to reject H0. Transformed residuals are homoscedastic!")
        
    # Generate Transformed OLS Diagnostic Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Residuals vs Fitted
    sns.scatterplot(
        x=y_hat_trans[sample_indices],
        y=residuals_trans[sample_indices],
        ax=axes[0],
        alpha=0.25,
        edgecolor='none',
        color='#9467bd'
    )
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Transformed Residuals vs Fitted (Subsampled 10k)')
    axes[0].set_xlabel('Transformed Fitted Values ($y_{hat}$)')
    axes[0].set_ylabel('Transformed Residuals ($e$)')
    
    # Plot 2: Normal Q-Q Plot
    # Compute new studentized residuals for transformed model
    SSRES_t = np.sum(residuals_trans ** 2)
    sigma_hat_t = np.sqrt(SSRES_t / (n - r_plus_1))
    studentized_residuals_trans = residuals_trans / (sigma_hat_t * np.sqrt(1.0 - h))
    
    sorted_stud_res_t = np.sort(studentized_residuals_trans)
    
    sns.scatterplot(
        x=theoretical_quantiles[sample_indices],
        y=sorted_stud_res_t[sample_indices],
        ax=axes[1],
        alpha=0.25,
        edgecolor='none',
        color='#d62728'
    )
    # Fit line through percentiles
    y25_t, y75_t = np.percentile(sorted_stud_res_t, [25, 75])
    slope_t = (y75_t - y25_t) / (x75 - x25)
    intercept_t = y25_t - slope_t * x25
    axes[1].plot(theoretical_quantiles, slope_t * theoretical_quantiles + intercept_t, color='red', linestyle='-', linewidth=1.5)
    
    axes[1].set_title('Transformed Normal Q-Q Plot')
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Studentized Residuals')
    
    plt.tight_layout()
    plt.savefig('plots/transformed_diagnostics.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/transformed_diagnostics.png')
    
    # Normality: Shapiro-Wilk Test (on a subsample of 5000 residuals due to large N sensitivity and limits)
    print("\nRunning Shapiro-Wilk Normality Test on a random subsample of 5000 residuals (Transformed Model, lambda=0.5)...")
    np.random.seed(42)
    subsample_res_t = np.random.choice(results_trans.resid, 5000, replace=False)
    shapiro_stat_t, shapiro_p_t = stats.shapiro(subsample_res_t)
    print(f"Shapiro-Wilk W-statistic (Transformed): {shapiro_stat_t:.6f}")
    print(f"Shapiro-Wilk p-value (Transformed): {shapiro_p_t:.8e}")
    if shapiro_p_t < 0.05:
        print("Result: Reject H0 (Normality). Transformed residuals are STILL NOT perfectly normally distributed, but normal approximation is vastly improved.")
    else:
        print("Result: Fail to reject H0 (Normality). Transformed residuals are approximately normal.")
    
    # -------------------------------------------------------------------------
    # STEP 6: NESTED MODEL COMPARISON & SELECTION (F-Test / ANOVA / VIF)
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Model Selection, Nested F-Test & VIF ---")
    
    # Backward Elimination based on AIC / BIC on the transformed model
    current_features = Z.columns.tolist().copy()
    # Keep intercept
    if 'Intercept' in current_features:
        current_features.remove('Intercept')
        
    print("Running Backward Elimination based on p-values (threshold = 0.05)...")
    step = 1
    while len(current_features) > 0:
        # Fit OLS
        Z_sub = Z[['Intercept'] + current_features]
        model_step = sm.OLS(Y_trans, Z_sub)
        res_step = model_step.fit()
        
        # Get p-values excluding Intercept
        p_values = res_step.pvalues.drop('Intercept')
        max_p = p_values.max()
        max_var = p_values.idxmax()
        
        if max_p > 0.05:
            print(f"Step {step}: Dropping '{max_var}' (p-value = {max_p:.4f}), AIC = {res_step.aic:.2f}, BIC = {res_step.bic:.2f}")
            current_features.remove(max_var)
            step += 1
        else:
            print(f"Backward Elimination completed. Final features to keep: {current_features}")
            print(f"Final Model AIC = {res_step.aic:.2f}, BIC = {res_step.bic:.2f}")
            break
            
    # Formal Nested F-test (ANOVA)
    # Compare Full Model vs Reduced Model (excluding Gender and all its interaction terms)
    gender_related = ['Gender', 'Gender_x_Seniority'] + [c for c in Z.columns if 'Gender_x_Job' in c]
    reduced_features = [c for c in Z.columns if c not in gender_related]
    
    Z_full = Z.values
    Z_reduced = Z[reduced_features].values
    
    # Fit full and reduced models on transformed Y
    res_full = sm.OLS(Y_trans, Z).fit()
    res_reduced = sm.OLS(Y_trans, Z[reduced_features]).fit()
    
    SSRES_full = np.sum(res_full.resid ** 2)
    SSRES_reduced = np.sum(res_reduced.resid ** 2)
    
    p_full = Z_full.shape[1]
    p_reduced = Z_reduced.shape[1]
    q = p_full - p_reduced  # number of restrictions
    
    # F-statistic formula: F = ((SSRES_red - SSRES_full)/q) / (SSRES_full / (n - p_full))
    F_stat = ((SSRES_reduced - SSRES_full) / q) / (SSRES_full / (n - p_full))
    F_pvalue = stats.f.sf(F_stat, q, n - p_full)
    
    print("\nNested F-Test (ANOVA): Full vs Reduced Model (No Gender Info)")
    print(f"Restrictions (coefficients tested for 0): {q} ({gender_related})")
    print(f"Full Model SSRES: {SSRES_full:,.4f} (Parameters: {p_full})")
    print(f"Reduced Model SSRES: {SSRES_reduced:,.4f} (Parameters: {p_reduced})")
    print(f"Nested F-Statistic: {F_stat:.4f}")
    print(f"F-Test p-value: {F_pvalue:.8e}")
    if F_pvalue < 0.05:
        print("Result: Strongly Reject H0! Gender pay gap and its interactions are highly statistically significant.")
    else:
        print("Result: Fail to reject H0. No statistical difference when removing gender indicators.")
        
    # Variance Inflation Factor (VIF)
    print("\nCalculating Variance Inflation Factors (VIF) for all covariates...")
    # Skip Intercept for VIF
    covariates_only = Z.drop(columns=['Intercept'])
    vif_data = pd.DataFrame()
    vif_data["Covariate"] = covariates_only.columns
    
    # statsmodels VIF takes values of dataframe and col index
    vifs = []
    for i in range(covariates_only.shape[1]):
        vif_val = variance_inflation_factor(covariates_only.values, i)
        vifs.append(vif_val)
        
    vif_data["VIF"] = vifs
    print(vif_data.sort_values(by="VIF", ascending=False))
    
    # -------------------------------------------------------------------------
    # STEP 7: MACHINE LEARNING & INFERENZA CAUSALE (Senza Seniority)
    # -------------------------------------------------------------------------
    print("\n--- Step 7: Advanced Machine Learning & Causal Inference ---")
    
    # 1. Prepare ML Feature Matrix (Strictly excluding Seniority)
    ml_features = year_dummies.columns.tolist() + job_dummies.columns.tolist()
    X_control = Z[ml_features].copy()
    T_treatment = final_df['Gender'].copy()
    Y_target = final_df['TotalPay'].copy()
    
    X_full = X_control.copy()
    X_full['Gender'] = T_treatment
    
    # Train-Test Split (80/20)
    X_train_full, X_test_full, Y_train, Y_test, T_train, T_test, X_train_ctrl, X_test_ctrl = train_test_split(
        X_full, Y_target, T_treatment, X_control, test_size=0.2, random_state=42
    )
    
    # --- Percorso A: XGBoost e SHAP ---
    print("\n--- Path A: XGBoost & SHAP ---")
    # Fit XGBoost on original scale
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_full, Y_train)
    
    y_pred_xgb = xgb_model.predict(X_test_full)
    rmse_xgb = np.sqrt(mean_squared_error(Y_test, y_pred_xgb))
    r2_xgb = r2_score(Y_test, y_pred_xgb)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_full)
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_full, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/shap_summary.png')
    
    # SHAP Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_full, plot_type="bar", show=False, color='#1f77b4')
    plt.title('SHAP Feature Importance (Bar)')
    plt.tight_layout()
    plt.savefig('plots/shap_bar.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/shap_bar.png')
    
    # XGBoost: Predicted vs Actual Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=Y_test, y=y_pred_xgb, alpha=0.3, edgecolor='none', color='#2ca02c')
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
    plt.xlabel('Salario Reale ($)')
    plt.ylabel('Salario Predetto da XGBoost ($)')
    plt.title('XGBoost: Previsto vs Reale (Out-of-Sample)')
    plt.tight_layout()
    plt.savefig('plots/xgb_pred_vs_actual.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/xgb_pred_vs_actual.png')

    
    

    # --- Percorso B: Double Machine Learning (DML) ---
    print("\n--- Path B: Double Machine Learning (DML) ---")
    
    # Nuisance Model 1: E[Y|X] -> Regression
    model_Y = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    # Nuisance Model 2: E[T|X] -> Classification
    model_T = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Computing out-of-fold residuals for DML...")
    E_Y_X_train = cross_val_predict(model_Y, X_train_ctrl, Y_train, cv=cv, n_jobs=-1)
    E_T_X_train = cross_val_predict(model_T, X_train_ctrl, T_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Orthogonalization
    Y_tilde = Y_train - E_Y_X_train
    T_tilde = T_train - E_T_X_train
    
    # Final Causal Inference via OLS
    dml_ols = sm.OLS(Y_tilde, T_tilde).fit()
    theta_hat = dml_ols.params.iloc[0]
    p_val_theta = dml_ols.pvalues.iloc[0]
    conf_int = dml_ols.conf_int(alpha=0.05).iloc[0].values
    
    print(f"Estimated Average Treatment Effect (ATE) of Gender on Salary: {theta_hat:.2f} $")
    print(f"p-value: {p_val_theta:.4e}")
    print(f"95% Confidence Interval: [{conf_int[0]:.2f}, {conf_int[1]:.2f}]")
    
    # Out-of-sample evaluation of nuisance models
    model_Y.fit(X_train_ctrl, Y_train)
    y_pred_dml_nuisance = model_Y.predict(X_test_ctrl)
    rmse_dml_nuisance = np.sqrt(mean_squared_error(Y_test, y_pred_dml_nuisance))
    r2_dml_nuisance = r2_score(Y_test, y_pred_dml_nuisance)
    
    # --- Final Out-Of-Sample Comparison with OLS (Box-Cox) ---
    print("\n--- Out-of-Sample Metrics Comparison ---")
    
    Z_train_full = Z.iloc[X_train_full.index].copy()
    Z_test_full = Z.iloc[X_test_full.index].copy()
    Y_trans_train = Y_trans[X_train_full.index]
    
    ols_train = sm.OLS(Y_trans_train, Z_train_full).fit()
    y_trans_pred_ols = ols_train.predict(Z_test_full)
    
    # Invert Box-Cox robustly
    y_pred_ols = np.where(y_trans_pred_ols < -1/opt_lambda, 0, inv_boxcox(y_trans_pred_ols, opt_lambda))
    
    rmse_ols = np.sqrt(mean_squared_error(Y_test, y_pred_ols))
    r2_ols = r2_score(Y_test, y_pred_ols)
    
    print(f"{'Model':<30} | {'RMSE ($)':<15} | {'R^2':<10}")
    print("-" * 60)
    print(f"{'OLS Box-Cox (lambda=0.5)':<30} | {rmse_ols:<15.2f} | {r2_ols:<10.4f}")
    print(f"{'XGBoost (Original Scale)':<30} | {rmse_xgb:<15.2f} | {r2_xgb:<10.4f}")
    print(f"{'DML Nuisance (E[Y|X])':<30} | {rmse_dml_nuisance:<15.2f} | {r2_dml_nuisance:<10.4f}")
    
    print("\nMachine Learning phase complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()

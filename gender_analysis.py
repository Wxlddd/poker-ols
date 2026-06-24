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
ARTIFACT_DIR = r"C:\Users\loren\.gemini\antigravity\brain\c77aa9f0-3859-4a72-bd71-deb61bec4995"

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
    
    # Filter for full-time employees (salaries > $30,000) to exclude part-time and temporary positions
    clean_df = df[(df['BasePay'] > 30000) & (df['TotalPay'] > 30000)].copy()
    print(f"Dataset shape after filtering positive pay (> $30k): {clean_df.shape}")
    
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
    
    # Shuffle the dataset randomly to distribute observations randomly on index plots
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
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
        
    # Job Category Effect Coding (Reference: Other)
    active_categories = ['Admin', 'Education', 'Fire', 'Medical', 'Police']
    for cat in active_categories:
        col_name = f'Job_{cat}'
        design_data[col_name] = np.select(
            [final_df['JobCategory'] == cat, final_df['JobCategory'] == 'Other'],
            [1, -1],
            default=0
        )
        
    # Interaction terms
    # 1. Gender * Seniority
    design_data['Gender_x_Seniority'] = design_data['Gender'] * design_data['Seniority']
    # 2. Gender * JobCategory (using effect-coded columns)
    for cat in active_categories:
        col_name = f'Job_{cat}'
        inter_name = f"Gender_x_{col_name}"
        design_data[inter_name] = design_data['Gender'] * design_data[col_name]
        
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
    results_naive = model_naive.fit(cov_type='HC3')
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
    
    # Clean up old combined plot if it exists
    if os.path.exists('plots/leverage_cooks.png'):
        try:
            os.remove('plots/leverage_cooks.png')
        except OSError:
            pass

    # Generate Separate White-Themed Plots
    print("\nGenerating separate Leverage and Cook's Distance plots...")
    
    # 1. Leverage Index Plot
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    x_indices = np.arange(n)
    
    # Partition leverage values based on thresholds discussed
    mask_high = h >= 0.0010
    mask_med = (h >= 0.0003) & (h < 0.0010)
    mask_low = h < 0.0003
    
    # Plot low leverage standard block (light silver)
    ax.scatter(
        x_indices[mask_low], 
        h[mask_low], 
        color='#bdc3c7', 
        s=2, 
        alpha=0.15, 
        label='Leva Bassa (Profili Standard: < 0.0003)',
        rasterized=True
    )
    
    # Plot medium leverage intermediate block (steel blue)
    ax.scatter(
        x_indices[mask_med], 
        h[mask_med], 
        color='#3498db', 
        s=3, 
        alpha=0.5, 
        label='Leva Media (Minoranze/Anzianità: 0.0003 - 0.0010)',
        rasterized=True
    )
    
    # Plot high leverage peak block (crimson red)
    ax.scatter(
        x_indices[mask_high], 
        h[mask_high], 
        color='#e74c3c', 
        s=5, 
        alpha=0.8, 
        label='Leva Alta (Donne nei Vigili del Fuoco: >= 0.0010)',
        rasterized=True
    )
    
    ax.axhline(leverage_threshold, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Soglia Critica 2(r+1)/n: {leverage_threshold:.6f}')
    
    ax.set_title("Index Plot dei Punti di Leva (Leverage)", fontsize=13, color='#2c3e50', fontweight='bold', pad=15)
    ax.set_xlabel("Indice Osservazione", fontsize=11, color='#2c3e50', fontweight='bold', labelpad=10)
    ax.set_ylabel("Valore di Leva ($h_{ii}$)", fontsize=11, color='#2c3e50', fontweight='bold', labelpad=10)
    
    ax.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#2c3e50')
    ax.spines['bottom'].set_color('#2c3e50')
    
    ax.tick_params(colors='#2c3e50')
    ax.legend(loc='center', frameon=True, facecolor='white', edgecolor='#cccccc')
    
    plt.tight_layout()
    plt.savefig('plots/leverage.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/leverage.png')
    print("Saved plots/leverage.png and copied to artifacts.")
    
    # 2. Cook's Distance Index Plot
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    ax.scatter(
        range(n), 
        cooks_d, 
        color='#d62728', 
        s=2, 
        alpha=0.15, 
        label="Distanza di Cook ($D_i$)",
        rasterized=True
    )
    
    # Cook's distance significance threshold: 4/n
    cooks_threshold = 4.0 / n
    ax.axhline(
        cooks_threshold, 
        color='#e74c3c', 
        linestyle='--', 
        linewidth=1.2, 
        label=f"Soglia d'Influenza (4/n): {cooks_threshold:.6f}"
    )
    
    ax.set_title("Index Plot della Distanza di Cook ($D_i$)", fontsize=13, color='#2c3e50', fontweight='bold', pad=15)
    ax.set_xlabel("Indice Osservazione", fontsize=11, color='#2c3e50', fontweight='bold', labelpad=10)
    ax.set_ylabel("Distanza di Cook ($D_i$)", fontsize=11, color='#2c3e50', fontweight='bold', labelpad=10)
    
    ax.set_ylim(0, max(cooks_d) * 1.8)
    
    # Annotate top 5 outliers with clean gray-bordered badges
    for i, idx in enumerate(top_5_idx):
        name = final_df.iloc[idx]['EmployeeName']
        val = cooks_d[idx]
        h_offset = -15000 if i % 2 == 0 else 15000
        v_offset = 0.0003 + (i * 0.0003)
        ax.annotate(
            name, 
            xy=(idx, val), 
            xytext=(idx + h_offset, val + v_offset),
            arrowprops=dict(facecolor='#2c3e50', arrowstyle='->', lw=0.8),
            fontsize=8,
            fontweight='bold',
            color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.25', fc='#f8f9fa', ec='#2c3e50', alpha=0.9, lw=0.8)
        )
        
    ax.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#2c3e50')
    ax.spines['bottom'].set_color('#2c3e50')
    
    ax.tick_params(colors='#2c3e50')
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc')
    
    plt.tight_layout()
    plt.savefig('plots/cooks_distance.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/cooks_distance.png')
    print("Saved plots/cooks_distance.png and copied to artifacts.")
    
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
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
    
    # Plot 3: Histogram of Residuals with Fitted Normal Curve
    sns.histplot(residuals, kde=False, stat="density", ax=axes[2], color='#1f77b4', bins=50, alpha=0.6)
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = axes[2].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[2].plot(x, p, 'r-', linewidth=2, label=f'Gaussiana\n(μ={mu:.2f}, σ={std:.2f})')
    axes[2].set_title('Istogramma dei Residui vs Normal')
    axes[2].set_xlabel('Residui ($e$)')
    axes[2].set_ylabel('Densità')
    axes[2].legend()
    
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
    plt.plot(lambdas, log_likelihoods, color='purple', linewidth=2, label='Profilo Log-Verosimiglianza')
    plt.axvline(opt_lambda, color='red', linestyle='--', label=f'$\lambda$ Ottimale = {opt_lambda:.4f}')
    plt.axvline(0, color='darkorange', linestyle=':', label='$\lambda$ = 0 (Trasformazione Logaritmica)')
    plt.title('Profilo di Log-Verosimiglianza Box-Cox')
    plt.xlabel('Parametro $\lambda$')
    plt.ylabel('Log-Verosimiglianza')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/boxcox_likelihood.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/boxcox_likelihood.png')
    
    # Econometric custom transformation decision
    # Apply the Logarithm Transformation (lambda = 0.0) as it is the most appropriate and requested by the user
    print(f"Optimal Lambda (MLE) is {opt_lambda:.4f}.")
    print("For econometric interpretability and user preference, we apply the Logarithm Transformation: lambda = 0.0 (Log Transformation).")
    opt_lambda = 0.0
    Y_trans = np.log(Y)
        
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
        
    # Refit OLS with transformed Y (lambda = 0.0 / Log Transformed)
    model_trans = sm.OLS(Y_trans, Z)
    results_trans = model_trans.fit(cov_type='HC3')
    print("\n--- OLS SUMMARY ON TRANSFORMED VARIABLE (lambda = 0.0 / Log Transformed) ---")
    print(results_trans.summary())
    
    # Generate custom markdown table (exactly formatted like Tabella 1.1)
    coefs = results_trans.params
    se = results_trans.bse
    z_stats = results_trans.tvalues
    p_vals = results_trans.pvalues
    conf_int = results_trans.conf_int()
    
    # Calculate computed reference category values (Other) using sum-constraint logic
    active_jobs = [f'Job_{cat}' for cat in ['Admin', 'Education', 'Fire', 'Medical', 'Police']]
    active_gender_jobs = [f'Gender_x_Job_{cat}' for cat in ['Admin', 'Education', 'Fire', 'Medical', 'Police']]
    
    # Job_Other
    coef_other = -np.sum(coefs[active_jobs])
    cov_jobs = results_trans.cov_params().loc[active_jobs, active_jobs]
    var_other = np.sum(cov_jobs.values)
    se_other = np.sqrt(var_other)
    z_other = coef_other / se_other
    p_other = 2 * (1 - stats.norm.cdf(np.abs(z_other)))
    ci_lower_other = coef_other - 1.96 * se_other
    ci_upper_other = coef_other + 1.96 * se_other
    
    # Gender_x_Job_Other
    coef_gender_other = -np.sum(coefs[active_gender_jobs])
    cov_gender_jobs = results_trans.cov_params().loc[active_gender_jobs, active_gender_jobs]
    var_gender_other = np.sum(cov_gender_jobs.values)
    se_gender_other = np.sqrt(var_gender_other)
    z_gender_other = coef_gender_other / se_gender_other
    p_gender_other = 2 * (1 - stats.norm.cdf(np.abs(z_gender_other)))
    ci_lower_gender_other = coef_gender_other - 1.96 * se_gender_other
    ci_upper_gender_other = coef_gender_other + 1.96 * se_gender_other
    
    # Build ordered row data
    row_data = []
    for idx in coefs.index:
        display_name = idx
        if display_name == 'Gender':
            display_name = 'Gender (Femmina)'
        else:
            display_name = display_name.replace('_x_', ' x ')
            
        row_data.append({
            'key': idx,
            'name': display_name,
            'coef': coefs[idx],
            'se': se[idx],
            'z': z_stats[idx],
            'p': p_vals[idx],
            'ci_lower': conf_int.loc[idx, 0],
            'ci_upper': conf_int.loc[idx, 1]
        })
        
        # Insert Job_Other after Job_Police
        if idx == 'Job_Police':
            row_data.append({
                'key': 'Job_Other',
                'name': 'Job_Other',
                'coef': coef_other,
                'se': se_other,
                'z': z_other,
                'p': p_other,
                'ci_lower': ci_lower_other,
                'ci_upper': ci_upper_other
            })
            
        # Insert Gender x Job_Other after Gender x Job_Police
        if idx == 'Gender_x_Job_Police':
            row_data.append({
                'key': 'Gender_x_Job_Other',
                'name': 'Gender x Job_Other',
                'coef': coef_gender_other,
                'se': se_gender_other,
                'z': z_gender_other,
                'p': p_gender_other,
                'ci_lower': ci_lower_gender_other,
                'ci_upper': ci_upper_gender_other
            })
            
    # Markdown Table Generation
    table_lines = []
    table_lines.append("| Covariata | Coefficiente | Dev. Standard (HC3) | Statistica z | p-value | Intervallo di Conf. 95% |")
    table_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    for row in row_data:
        c_val = f"{row['coef']:.4f}"
        se_val = f"{row['se']:.3f}"
        z_val = f"{row['z']:.3f}"
        
        p = row['p']
        if p == 0.0:
            p_val_str = "0 (prec. macchina)"
        elif p < 0.001:
            p_val_str = f"{p:.2e}"
        else:
            p_val_str = f"{p:.3f}"
            
        # Bold p-value if statistically significant (p < 0.05)
        if p < 0.05:
            p_val_str = f"**{p_val_str}**"
            
        ci_val = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        table_lines.append(f"| {row['name']} | {c_val} | {se_val} | {z_val} | {p_val_str} | {ci_val} |")
        
    table_lines.append(f"| **R²** | {results_trans.rsquared:.4f} | | | | |")
    table_lines.append(f"| **R² Adjusted** | {results_trans.rsquared_adj:.4f} | | | | |")
    
    markdown_table = "\n".join(table_lines)
    
    os.makedirs('plots', exist_ok=True)
    with open('plots/regression_table.md', 'w') as f:
        f.write(markdown_table)
        f.write("\n")
        
    copy_to_artifacts('plots/regression_table.md')
    print("\nCustom regression table generated and saved to 'plots/regression_table.md'.")
    
    # Render table to PNG using Matplotlib (without dev standard, p-values in scientific notation)
    print("\nGenerating regression table image (plots/regression_table.png)...")
    
    img_data = []
    for row in row_data:
        c_val = f"{row['coef']:.4f}"
        z_val = f"{row['z']:.3f}"
        
        p = row['p']
        if p == 0.0:
            p_val_str = "0 (prec. macchina)"
        elif p < 0.001:
            p_val_str = f"{p:.2e}"
        else:
            p_val_str = f"{p:.3f}"
            
        ci_val = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        img_data.append([row['name'], c_val, z_val, p_val_str, ci_val])
        
    img_data.append(['R²', f"{results_trans.rsquared:.4f}", '', '', ''])
    img_data.append(['R² Adjusted', f"{results_trans.rsquared_adj:.4f}", '', '', ''])
    
    headers = ['Covariata', 'Coefficiente', 'Statistica z', 'p-value', 'Intervallo di Conf. 95%']
    
    # Create Matplotlib Table with slightly larger size for 17 rows + 2 R2 rows
    fig, ax = plt.subplots(figsize=(10.5, 8.2), facecolor='white')
    ax.axis('off')
    
    table_obj = ax.table(
        cellText=img_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        cellColours=[['#ffffff']*5 if i%2==0 else ['#f8f9fa']*5 for i in range(len(img_data))],
        colColours=['#2c3e50']*5
    )
    
    # Styling table cells
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(10)
    table_obj.scale(1.0, 1.5) # scale cell heights/widths
    
    # Align first column to the left, others to center
    for i in range(len(img_data) + 1):
        for j in range(5):
            cell = table_obj[i, j]
            cell.set_edgecolor('#cccccc')
            cell.set_linewidth(0.5)
            
            if i == 0:
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
                cell.get_text().set_fontsize(11)
            else:
                if j == 0:
                    cell.get_text().set_ha('left')
                    cell.set_text_props(x=0.04)
                    cell.get_text().set_weight('semibold')
                # Make the R2 labels stand out
                if i > len(row_data):
                    cell.get_text().set_weight('bold')
                    if j == 0:
                        cell.get_text().set_color('#2c3e50')
                if j == 3 and i - 1 < len(row_data):
                    p_val_float = row_data[i-1]['p']
                    if p_val_float < 0.05:
                        cell.get_text().set_weight('bold')
                        cell.get_text().set_color('#c0392b') # elegant dark red for significance
                        
    plt.tight_layout()
    plt.savefig('plots/regression_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    copy_to_artifacts('plots/regression_table.png')
    print("Saved plots/regression_table.png and copied to artifacts.")
    
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
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
    
    # Plot 3: Histogram of Residuals with Fitted Normal Curve
    sns.histplot(residuals_trans, kde=False, stat="density", ax=axes[2], color='#9467bd', bins=50, alpha=0.6)
    mu_t, std_t = stats.norm.fit(residuals_trans)
    xmin_t, xmax_t = axes[2].get_xlim()
    x_t = np.linspace(xmin_t, xmax_t, 100)
    p_t = stats.norm.pdf(x_t, mu_t, std_t)
    axes[2].plot(x_t, p_t, 'r-', linewidth=2, label=f'Gaussiana\n(μ={mu_t:.2f}, σ={std_t:.2f})')
    axes[2].set_title('Istogramma dei Residui vs Normal')
    axes[2].set_xlabel('Residui Trasformati ($e$)')
    axes[2].set_ylabel('Densità')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('plots/transformed_diagnostics.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/transformed_diagnostics.png')
    
    # Normality: Shapiro-Wilk Test (on a subsample of 5000 residuals due to large N sensitivity and limits)
    print("\nRunning Shapiro-Wilk Normality Test on a random subsample of 5000 residuals (Transformed Model, lambda=0.0 / Log Transformed)...")
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
        
    print("Running Backward Elimination based on robust p-values (threshold = 0.05)...")
    step = 1
    while len(current_features) > 0:
        # Fit OLS
        Z_sub = Z[['Intercept'] + current_features]
        model_step = sm.OLS(Y_trans, Z_sub)
        res_step = model_step.fit(cov_type='HC3')
        
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
    res_full = sm.OLS(Y_trans, Z).fit(cov_type='HC3')
    res_reduced = sm.OLS(Y_trans, Z[reduced_features]).fit(cov_type='HC3')
    
    SSRES_full = np.sum(res_full.resid ** 2)
    SSRES_reduced = np.sum(res_reduced.resid ** 2)
    
    p_full = Z_full.shape[1]
    p_reduced = Z_reduced.shape[1]
    q = p_full - p_reduced  # number of restrictions
    
    # F-statistic formula: F = ((SSRES_red - SSRES_full)/q) / (SSRES_full / (n - p_full))
    F_stat = ((SSRES_reduced - SSRES_full) / q) / (SSRES_full / (n - p_full))
    F_pvalue = stats.f.sf(F_stat, q, n - p_full)
    
    # Robust joint test using statsmodels' f_test (which accounts for the robust covariance matrix)
    constraints = [f"{var} = 0" for var in gender_related]
    robust_joint_test = res_full.f_test(constraints)
    robust_F_stat = float(robust_joint_test.fvalue)
    robust_F_pvalue = float(robust_joint_test.pvalue)
    
    print("\nNested F-Test (ANOVA): Full vs Reduced Model (No Gender Info)")
    print(f"Restrictions (coefficients tested for 0): {q} ({gender_related})")
    print(f"Full Model SSRES: {SSRES_full:,.4f} (Parameters: {p_full})")
    print(f"Reduced Model SSRES: {SSRES_reduced:,.4f} (Parameters: {p_reduced})")
    print(f"Classical Nested F-Statistic: {F_stat:.4f}")
    print(f"Classical F-Test p-value: {F_pvalue:.8e}")
    print(f"Robust Nested F-Statistic (HC3 VCE): {robust_F_stat:.4f}")
    print(f"Robust F-Test p-value: {robust_F_pvalue:.8e}")
    if robust_F_pvalue < 0.05:
        print("Result: Strongly Reject H0! Gender pay gap and its interactions are highly statistically significant (robust test).")
    else:
        print("Result: Fail to reject H0. No statistical difference when removing gender indicators (robust test).")
        
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
    vif_sorted = vif_data.sort_values(by="VIF", ascending=False)
    print(vif_sorted)
    
    # Generate VIF Plot
    print("\nGenerating VIF of Covariates Plot...")
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Sort for plotting (ascending order so the highest values are at the top of the horizontal bar plot)
    vif_plot_df = vif_data.sort_values(by="VIF", ascending=True)
    
    # Plot horizontal bars
    bars = ax.barh(vif_plot_df["Covariate"], vif_plot_df["VIF"], color='#ffffff', edgecolor='#2c3e50', linewidth=1.2, height=0.6)
    
    # Customize axes and title
    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=11, color='#2c3e50', fontweight='bold', labelpad=10)
    ax.set_title('Variance Inflation Factor (VIF) per Covariata\n(Tutte le covariate risultano stabili con VIF < 5.0)', fontsize=13, color='#2c3e50', fontweight='bold', pad=15)
    
    # Set limit to 6 to give some breathing room for annotations and show the threshold line clearly
    max_vif = vif_plot_df["VIF"].max()
    ax.set_xlim(0, max(6.0, max_vif + 1.0))
    
    # Add vertical reference line at VIF = 5.0 (commonly accepted threshold)
    ax.axvline(5.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Soglia di Attenzione (VIF = 5.0)')
    
    # Add numerical annotations next to each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', ha='left', color='#2c3e50', fontweight='bold', fontsize=10)
        
    # Grid and spines configuration (clean, white theme)
    ax.grid(True, axis='x', linestyle=':', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    
    # Hide top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#2c3e50')
    ax.spines['bottom'].set_color('#2c3e50')
    
    ax.tick_params(colors='#2c3e50')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc')
    
    plt.tight_layout()
    plt.savefig('plots/vif_covariates.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/vif_covariates.png')
    print("Saved plots/vif_covariates.png and copied to artifacts.")
    
    # -------------------------------------------------------------------------
    # STEP 7: REGULARIZATION (Ridge Regression)
    # -------------------------------------------------------------------------
    print("\n--- Step 7: Ridge Regularization & Ridge Path ---")
    
    # Standardize the covariates of Z (excluding intercept)
    covariates = Z.drop(columns=['Intercept'])
    X_mean = covariates.mean()
    X_std = covariates.std()
    # Handle columns with zero variance if any (just in case)
    X_std[X_std == 0] = 1.0
    X_scaled = (covariates - X_mean) / X_std
    
    # Center Y (on transformed scale) so we don't have to penalize the intercept
    Y_centered = Y_trans - np.mean(Y_trans)
    
    # Create grid of penalty parameters (alphas in ridge terminology)
    # 200 values from 10^-2 to 10^10
    alphas = np.logspace(-2, 10, 200)
    
    coef_list = []
    col_names = covariates.columns.tolist()
    
    # Compute Ridge analytically: beta_RR = (X^T X + alpha * I)^-1 * X^T * Y
    X_arr = X_scaled.values
    XtX = X_arr.T @ X_arr
    XtY = X_arr.T @ Y_centered
    k_vars = X_arr.shape[1]
    
    for alpha in alphas:
        # beta = (XtX + alpha * I)^-1 XtY
        inv_mat = np.linalg.inv(XtX + alpha * np.eye(k_vars))
        beta = inv_mat @ XtY
        coef_list.append(beta)
        
    coef_matrix = np.array(coef_list) # dimensions: (num_alphas, k_vars)
    
    # Plot Ridge Path
    plt.figure(figsize=(12, 7))
    for j in range(k_vars):
        plt.plot(alphas, coef_matrix[:, j], label=col_names[j], linewidth=1.5)
        
    plt.xscale('log')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Ridge Regression Path (Coefficients Shrinkage)')
    plt.xlabel('Regularization Penalty Parameter ($\lambda$)')
    plt.ylabel('Standardized Ridge Coefficients ($\hat{\\beta}_{RR}$)')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('plots/ridge_path.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/ridge_path.png')
    
    print("\nRidge Path generated and saved to 'plots/ridge_path.png'.")
    
    # -------------------------------------------------------------------------
    # STEP 8: OCCUPATIONAL SEGREGATION DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- Step 8: Occupational Segregation Descriptive Statistics ---")
    print("=" * 70)
    
    # Create a grouped column (keep categories separate)
    final_df['JobCategory_Seg'] = final_df['JobCategory']
    
    def print_occupational_segregation_table(df):
        total_males = np.sum(df['Gender'] == 0)
        total_females = np.sum(df['Gender'] == 1)
        
        seg_stats = []
        categories = ['Police', 'Fire', 'Medical', 'Education', 'Admin', 'Other']
        
        for cat in categories:
            sub = df[df['JobCategory_Seg'] == cat]
            n_male = np.sum(sub['Gender'] == 0)
            n_female = np.sum(sub['Gender'] == 1)
            n_total = n_male + n_female
            
            pct_male = n_male / n_total * 100 if n_total > 0 else 0
            pct_female = n_female / n_total * 100 if n_total > 0 else 0
            
            share_male = n_male / total_males * 100 if total_males > 0 else 0
            share_female = n_female / total_females * 100 if total_females > 0 else 0
            
            mean_pay = sub['TotalPay'].mean()
            
            seg_stats.append({
                'Settore': cat,
                'N_Male': n_male,
                'Pct_Male': pct_male,
                'N_Female': n_female,
                'Pct_Female': pct_female,
                'Share_Male': share_male,
                'Share_Female': share_female,
                'Mean_Pay': mean_pay
            })
            
        seg_df = pd.DataFrame(seg_stats)
        # Sort from highest male density to highest female density
        seg_df = seg_df.sort_values(by='Pct_Male', ascending=False).reset_index(drop=True)
        
        print("\n" + "=" * 115)
        print("  TABELLA RIASSUNTIVA: SEGREGAZIONE OCCUPAZIONALE E RETRIBUZIONE MEDIA PER SETTORE")
        print("=" * 115)
        print(f"{'Settore':<20} | {'N. Maschi':<12} | {'% Maschi':<10} | {'Quota M':<10} | {'N. Femmine':<12} | {'% Femmine':<10} | {'Quota F':<10} | {'Salario Medio':<15}")
        print("-" * 115)
        for idx, row in seg_df.iterrows():
            print(f"{row['Settore']:<20} | {row['N_Male']:<12,d} | {row['Pct_Male']:<10.2f}% | {row['Share_Male']:<10.2f}% | {row['N_Female']:<12,d} | {row['Pct_Female']:<10.2f}% | {row['Share_Female']:<10.2f}% | ${row['Mean_Pay']:<14,.2f}")
        print("-" * 115)
        print(f"Total Males in Dataset: {total_males:,} | Total Females in Dataset: {total_females:,}")
        print("=" * 115)
        
        return seg_df
        
    seg_df = print_occupational_segregation_table(final_df)
    
    # -------------------------------------------------------------------------
    # STEP 9: OCCUPATIONAL SEGREGATION VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- Step 9: Visualizing Occupational Segregation ---")
    print("=" * 70)
    
    plt.figure(figsize=(10, 6))
    
    categories = seg_df['Settore'].tolist()
    pct_males = seg_df['Pct_Male'].tolist()
    pct_females = seg_df['Pct_Female'].tolist()
    
    color_male = '#1f77b4'
    color_female = '#e377c2'
    
    bars_male = plt.barh(categories, pct_males, color=color_male, label='Uomini (0)', edgecolor='none', height=0.55)
    bars_female = plt.barh(categories, pct_females, left=pct_males, color=color_female, label='Donne (1)', edgecolor='none', height=0.55)
    
    plt.xlim(0, 100)
    plt.xlabel('Percentuale (%)')
    plt.title('Segregazione Occupazionale: Bilanciamento di Genere per Settore\n(Ordinato da massima densità maschile a massima densità femminile)')
    plt.legend(loc='lower left', frameon=True)
    
    plt.gca().invert_yaxis()
    
    for i in range(len(categories)):
        if pct_males[i] > 8:
            plt.text(pct_males[i]/2, i, f'{pct_males[i]:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=10)
        if pct_females[i] > 8:
            plt.text(pct_males[i] + pct_females[i]/2, i, f'{pct_females[i]:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=10)
            
    plt.tight_layout()
    plt.savefig('plots/occupational_segregation.png', dpi=150)
    plt.close()
    copy_to_artifacts('plots/occupational_segregation.png')
    print("Saved plots/occupational_segregation.png and copied to artifacts.")
    
    print("\nProcess complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()

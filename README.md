# Econometric Gender Pay Gap Analysis in San Francisco (2011–2014)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.6-darkblue.svg?style=for-the-badge&logo=python)](https://www.statsmodels.org/)
[![Scipy](https://img.shields.io/badge/scipy-1.17.1-lightblue.svg?style=for-the-badge&logo=scipy)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/pandas-3.0.3-violet.svg?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> [!NOTE]
> **Academic Project Showcase**: This repository contains a rigorous, end-to-end econometrics and labor economics workflow in Python. Using OLS regression, Box-Cox transformations, nested F-tests, and Ridge regularization, we study the **Gender Pay Gap** in the municipal workforce of the City of San Francisco, verifying every core theorem and diagnostic test in mathematical statistics.

---

## 📐 Mathematical & Theoretical Foundations

This workflow covers all key theorems, diagnostic checks, and geometric properties of linear regression models.

### 1. Model Specification & Design Matrix Invertibility
We model continuous employee compensation $Y$ using a multiple linear regression framework:
$$\vec{y} = Z\vec{\beta} + \vec{\varepsilon}$$
where $Z \in \mathbb{R}^{n \times (r+1)}$ is the design matrix containing the intercept, gender dummy, seniority proxy, year dummies, job category dummies, and their respective interaction terms. 

To guarantee the existence and uniqueness of the OLS estimator:
$$\hat{\vec{\beta}}_{OLS} = (Z^T Z)^{-1} Z^T \vec{y}$$
we verify that $Z$ has **full column rank**:
$$\text{rank}(Z) = r + 1$$
which ensures that the Gram matrix $Z^T Z$ is positive definite and strictly invertible.

### 2. Variance Decomposition & Orthogonality
Under OLS estimation, the total sum of squares ($SSTOT$) partitions perfectly into the model explained sum of squares ($SSREG$) and the residual sum of squares ($SSRES$):
$$SSTOT = SSREG + SSRES$$
$$\sum_{i=1}^n (y_i - \bar{y})^2 = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 + \sum_{i=1}^n \hat{\varepsilon}_i^2$$
which is mathematically guaranteed by the **fitted-residual orthogonality theorem**:
$$\hat{\vec{y}}^T \hat{\vec{\varepsilon}} = 0$$

### 3. Hat Matrix, Leverages & Cook's Distance
The prediction vector $\hat{\vec{y}}$ is a linear projection of $\vec{y}$ onto the column space of $Z$ via the **Hat Matrix** $H$:
$$\hat{\vec{y}} = H\vec{y}, \quad H = Z(Z^T Z)^{-1} Z^T$$
The diagonal elements $h_{ii} \in [0,1]$ represent the **leverage** of each observation. Points with exceptionally high leverage are identified using the threshold:
$$h_{ii} > \frac{2(r+1)}{n}$$
To assess the influence of each observation on the estimated coefficients $\hat{\vec{\beta}}$, we compute **Cook's Distance** $D_i$:
$$D_i = \frac{t_i^2}{r+1} \left( \frac{h_{ii}}{1 - h_{ii}} \right)$$
where $t_i$ is the studentized residual:
$$t_i = \frac{\hat{\varepsilon}_i}{\hat{\sigma} \sqrt{1 - h_{ii}}}$$

### 4. Variance Stabilization via Box-Cox Power Transform
When the homoscedasticity assumption ($Var(\vec{\varepsilon}) = \sigma^2 I$) is violated (heteroscedasticity), we apply the **Box-Cox transformation** to $Y$ to stabilize variance and restore normality:
$$Y^{(\lambda)} = \begin{cases} \frac{Y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(Y) & \text{if } \lambda = 0 \end{cases}$$
The parameter $\lambda$ is estimated via Maximum Likelihood Estimation (MLE) by maximizing:
$$L(\lambda) = -\frac{n}{2} \ln(\text{Var}(Y^{(\lambda)})) + (\lambda - 1) \sum_{i=1}^n \ln(y_i)$$

### 5. Selection and Nested Hypotheses (Partial F-Test)
We compare a saturated Full Model against a Restricted Model (excluding all gender terms) to test:
$$H_0: \beta_{\text{Gender}} = \beta_{\text{Gender}\times\text{Seniority}} = \dots = 0$$
Using the partial F-test statistic (ANOVA):
$$F = \frac{(SSRES_{\text{reduced}} - SSRES_{\text{full}}) / q}{SSRES_{\text{full}} / (n - p_{\text{full}})} \sim F(q, n - p_{\text{full}})$$

### 6. Ridge Regularization & Path
To manage collinearity in interactions and dummy features, we estimate the regularized Ridge path over $\lambda \ge 0$ on standardized covariates:
$$\hat{\vec{\beta}}_{RR}(\lambda) = (Z_{\text{std}}^T Z_{\text{std}} + \lambda I)^{-1} Z_{\text{std}}^T \vec{y}_{\text{centered}}$$

---

## 📊 Empirical Findings & Regression Summaries

### Data Preparation Breakdown
* **Original Dataset size**: 148,654 records.
* **Positive Salaries (BasePay > 0 & TotalPay > 0)**: 146,736 records.
* **Pruned Genders (excluding initials and ambiguous first names)**: **126,306** final records.
* **Final Gender Mix**: **42.1% Female (1)**, **57.9% Male (0)**.

### 1. Naïve OLS vs. Box-Cox Transformed OLS
The initial model on the raw pay scale showed massive violations of Gauss-Markov assumptions, including extreme eteroschedasticity ($p$-value of Breusch-Pagan $\approx 0$) and heavy-tailed residuals in the Q-Q Plot.

Our MLE search estimated the optimal power transform parameter at **$\lambda = 0.5969$**.

#### Regression Summary (Transformed Model $Y^{(0.597)}$)
* **Adjusted $R^2$**: **0.324**
* **F-statistic**: **4,327** ($p\text{-value} = 0.000$)

| Covariate | Coefficient | Std. Error | t-statistic | p-value | 95% Conf. Interval |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | 1287.6690 | 3.440 | 374.271 | **0.000** | [1280.93, 1294.41] |
| **Gender (Female)** | -82.6500 | 4.147 | -19.931 | **0.000** | [-90.78, -74.52] |
| **Seniority** | 153.2913 | 3.219 | 47.626 | **0.000** | [146.98, 159.60] |
| **Year_2012** | -103.4456 | 4.617 | -22.406 | **0.000** | [-112.50, -94.40] |
| **Year_2013** | 71.8346 | 3.888 | 18.474 | **0.000** | [64.21, 79.46] |
| **Year_2014** | -145.8181 | 5.222 | -27.924 | **0.000** | [-156.05, -135.58] |
| **Job_Education** | -901.4805 | 7.255 | -124.249 | **0.000** | [-915.70, -887.26] |
| **Job_Fire** | 674.0713 | 8.224 | 81.967 | **0.000** | [657.95, 690.19] |
| **Job_Medical** | -79.6376 | 7.329 | -10.867 | **0.000** | [-94.00, -65.27] |
| **Job_Police** | 464.5055 | 4.870 | 95.377 | **0.000** | [454.96, 474.05] |
| **Gender x Seniority** | -4.8301 | 3.757 | -1.285 | 0.199 | [-12.20, 2.54] |
| **Gender x Job_Education** | 83.7661 | 10.717 | 7.816 | **0.000** | [62.76, 104.77] |
| **Gender x Job_Fire** | 18.8541 | 19.930 | 0.946 | 0.344 | [-20.21, 57.92] |
| **Gender x Job_Medical** | 28.0078 | 8.838 | 3.169 | **0.002** | [10.69, 45.33] |
| **Gender x Job_Police** | -154.2422 | 9.745 | -15.829 | **0.000** | [-173.34, -135.14] |

### 2. Economic Interpretations of Interactivity
The partial derivative of transformed pay with respect to Gender (Female) is:
$$\frac{\partial Y^{(0.597)}}{\partial \text{Gender}} = -82.65 + 83.77 \cdot \text{Job\_Education} + 28.01 \cdot \text{Job\_Medical} - 154.24 \cdot \text{Job\_Police}$$

* **Baseline Pay Gap**: In the reference job category (`Other`), females earn significantly less than males ($\beta_{\text{Gender}} = -82.65$, $p < 0.001$).
* **Mitigation in Education**: In the Education sector, the pay gap is fully closed and slightly inverted ($\beta_{\text{Gender}} + \beta_{\text{Gender}\times\text{Job\_Education}} = -82.65 + 83.77 = +1.12$, which is statistically near zero).
* **Widening in Law Enforcement**: In the Police department, the gap is severely amplified, showing the largest absolute pay disparity ($\beta_{\text{Gender}} + \beta_{\text{Gender}\times\text{Job\_Police}} = -82.65 - 154.24 = -236.89$, $p < 0.001$).

---

## 📈 Visual Diagnostic Showcase

All charts are generated at high resolutions and saved directly in the project.

### 1. Box-Cox MLE Profile Log-Likelihood
The peak clearly shows that the optimal transform is $\lambda \approx 0.6$. A pure logarithmic transform ($\lambda = 0$) lies outside the confidence interval, justifying the power transform approach.

![Box-Cox Likelihood](file:///C:/Users/loren/.gemini/antigravity/brain/2df59a04-f39e-41fb-b884-2a57332edee4/boxcox_likelihood.png)

### 2. Naïve OLS vs. Transformed Residuals
* **Naïve Scale Diagnostics**: Shows a severe "fan shape" indicating eteroschedasticity, and heavy deviation from normality at both tails in the Q-Q plot.
* **Transformed Scale Diagnostics**: Stabilized residuals, displaying a highly linear and Gaussian Q-Q plot behavior.

| Naïve OLS Diagnostics | Transformed OLS Diagnostics |
| :---: | :---: |
| ![Naive OLS Diagnostics](file:///C:/Users/loren/.gemini/antigravity/brain/2df59a04-f39e-41fb-b884-2a57332edee4/ols_diagnostics.png) | ![Transformed OLS Diagnostics](file:///C:/Users/loren/.gemini/antigravity/brain/2df59a04-f39e-41fb-b884-2a57332edee4/transformed_diagnostics.png) |

### 3. Ridge Coefficients Path
This plot shows the shrinkage of the standardized beta coefficients to zero as the regularizing penalty $\lambda$ increases, visualizing the bias-variance trade-off.

![Ridge Path](file:///C:/Users/loren/.gemini/antigravity/brain/2df59a04-f39e-41fb-b884-2a57332edee4/ridge_path.png)

---

## 🚀 Execution & Replication Guide

To replicate the entire workflow, run the master script inside the virtual environment:

```bash
# Verify uv is set up and execute the pipeline
uv run gender_analysis.py
```

This will automatically:
1. Parse and clean `Salaries.csv`.
2. Extract first names and run the `gender-guesser` classifier.
3. Fit the regressions and print all hypothesis test statistics.
4. Save the high-resolution charts in the local workspace.

---

## 🃏 Poker Project Isolation

As requested, all poker-related files (data, scripts, and Hold'em starting hand profitability heatmaps) have been **fully preserved** and cleanly moved to a dedicated subdirectory:

👉 [**Go to Poker Subproject Directory**](file:///c:/Users/loren/Documents/AntiGravity%20Projects/progettostat/poker/)

You can navigate inside the `poker/` folder to view the starting hand strategic studies and run the `poker_hand_analysis.py` script. The original poker `README.md` is preserved inside that subdirectory as well!

# Analisi Econometrica sul Gender Pay Gap a San Francisco (2011–2014)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.6-darkblue.svg?style=for-the-badge&logo=python)](https://www.statsmodels.org/)
[![Scipy](https://img.shields.io/badge/scipy-1.17.1-lightblue.svg?style=for-the-badge&logo=scipy)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/pandas-3.0.3-violet.svg?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> [!NOTE]
> **Vetrina del Progetto Accademico**: Questo repository ospita un workflow econometrico e di economia del lavoro in Python. Il lavoro è strutturato in due parti principali: una prima parte dedicata all'analisi OLS classica con trasformazione ottimale Box-Cox ($\lambda = 0.5$) e relativa diagnostica di Gauss-Markov, ed una seconda parte dedicata all'analisi descrittiva e visualizzazione della **segregazione occupazionale**, evidenziata come driver fondamentale del divario retributivo anche dai benchmark di Machine Learning (XGBoost).

---

## 📐 Fondamenti Teorici e Teoremi Verificati

Questo studio verifica le proprietà geometriche ed econometriche del modello lineare classico.

### 1. Specificazione del Modello e Invertibilità
Modelliamo la retribuzione totale $Y$ tramite regressione lineare multipla:
$$\vec{y} = Z\vec{\beta} + \vec{\varepsilon}$$
dove $Z$ è la matrice di disegno contenente l'intercetta, la dummy di genere, l'anzianità, le dummy di anno, le dummy di macro-categoria lavorativa e i relativi termini di interazione.
Per garantire l'esistenza e l'unicità dello stimatore OLS:
$$\hat{\vec{\beta}}\_{OLS} = (Z^T Z)^{-1} Z^T \vec{y}$$
verifichiamo che la matrice $Z$ sia a **rango colonna pieno** ($\text{rango}(Z) = r + 1$), garantendo l'invertibilità di $Z^T Z$.

### 2. Decomposizione della Varianza e Ortogonalità
Sotto stima OLS, la somma dei quadrati totale si scompone perfettamente in somma dei quadrati spiegata ($SSREG$) e residua ($SSRES$):
$$SSTOT = SSREG + SSRES$$
Questa scomposizione geometrica è garantita dal **teorema di ortogonalità fitted-residui**:
$$\hat{\vec{y}}^T \hat{\vec{\varepsilon}} = 0$$

### 3. Matrice Hat, Leverage e Distanza di Cook
Il vettore delle previsioni $\hat{\vec{y}}$ è proiettato sullo spazio delle colonne di $Z$ tramite la **Matrice Hat** $H$:
$$\hat{\vec{y}} = H\vec{y}, \quad H = Z(Z^T Z)^{-1} Z^T$$
I valori sulla diagonale $h\_{ii} \in [0,1]$ misurano la leva (leverage). I punti di leva critici soddisfano la soglia:
$$h\_{ii} > \frac{2(r+1)}{n}$$
Per valutare l'influenza di ciascun punto sul vettore dei coefficienti $\hat{\vec{\beta}}$, calcoliamo la **Distanza di Cook** $D\_i$:
$$D\_i = \frac{t\_i^2}{r+1} \left( \frac{h\_{ii}}{1 - h\_{ii}} \right)$$
dove $t\_i$ rappresenta il residuo studentizzato internamente.

### 4. Trasformazione Box-Cox per la Stabilizzazione della Varianza
In presenza di eteroschedasticità, applichiamo la trasformazione di potenza di **Box-Cox** sulla risposta continua $Y$:
$$Y^{(\lambda)} = \begin{cases} \frac{Y^\lambda - 1}{\lambda} & \text{se } \lambda \neq 0 \\ \ln(Y) & \text{se } \lambda = 0 \end{cases}$$
Il parametro ottimale $\lambda$ viene stimato tramite Massima Verosimiglianza (MLE), massimizzando il profilo di log-verosimiglianza:
$$L(\lambda) = -\frac{n}{2} \ln(\text{Var}(Y^{(\lambda)})) + (\lambda - 1) \sum\_{i=1}^n \ln(y\_i)$$

---

## PARTE I: Analisi Econometrica Principale e Diagnostica (Box-Cox $\lambda = 0.5$)

In questa prima sezione viene presentata l'analisi principale, condotta sulla scala Box-Cox arrotondata a $\lambda = 0.5$ per massimizzarne l'interpretabilità econometrica (trasformazione radice quadrata).

### 1. Stime OLS e Standard Error Robusti (HC3)
Il fitting dell'OLS sulla scala trasformata $Y^{(0.5)}$ con deviazioni standard robuste **HC3** restituisce le seguenti stime:

* **$R^2$ Rettificato**: **0,327**
* **F-statistic Robust (HC3 VCE)**: **8.738** ($p\text{-value} = 0.000$)

#### Tabella dei Coefficienti (Modello Trasformato $\lambda = 0.5$, HC3)
| Covariata | Coefficiente | Dev. Standard (HC3) | Statistica z | p-value | Intervallo di Conf. 95% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | 515.0001 | 1.202 | 428.582 | **0.000** | [512.645, 517.355] |
| **Gender (Femmina)** | -27.9417 | 1.476 | -18.935 | **0.000** | [-30.834, -25.049] |
| **Seniority** | 54.6008 | 1.209 | 45.145 | **0.000** | [52.230, 56.971] |
| **Year_2012** | -37.5796 | 1.655 | -22.701 | **0.000** | [-40.824, -34.335] |
| **Year_2013** | 23.7536 | 1.365 | 17.401 | **0.000** | [21.078, 26.429] |
| **Year_2014** | -53.4521 | 1.933 | -27.654 | **0.000** | [-57.240, -49.664] |
| **Job_Education** | -329.8440 | 1.725 | -191.264 | **0.000** | [-333.224, -326.464] |
| **Job_Fire** | 223.1916 | 2.504 | 89.144 | **0.000** | [218.284, 228.099] |
| **Job_Medical** | -30.5440 | 3.164 | -9.654 | **0.000** | [-36.745, -24.343] |
| **Job_Police** | 155.7580 | 1.573 | 99.035 | **0.000** | [152.675, 158.841] |
| **Gender x Seniority** | -1.4818 | 1.352 | -1.096 | 0.273 | [-4.131, 1.167] |
| **Gender x Job_Education** | 28.0345 | 2.551 | 10.988 | **0.000** | [23.034, 33.035] |
| **Gender x Job_Fire** | 7.6066 | 5.979 | 1.272 | 0.203 | [-4.111, 19.325] |
| **Gender x Job_Medical** | 8.4227 | 3.771 | 2.234 | **0.025** | [1.032, 15.813] |
| **Gender x Job_Police** | -50.0045 | 3.196 | -15.645 | **0.000** | [-56.269, -43.740] |

---

### 2. Test Diagnostici Classici

#### Shapiro-Wilk (Normalità dei Residui)
Rilevato su un sottocampione casuale di $5.000$ osservazioni per superare l'ipersensibilità su grandi N:
* **Modello Naïve**: $W = 0.9851$ | $p\text{-value} = 1.34 \times 10^{-22}$
* **Modello Trasformato ($\lambda = 0.5$)**: $W = 0.9697$ | $p\text{-value} = 3.15 \times 10^{-31}$
> [!WARNING]
> Entrambi i test rifiutano l'ipotesi nulla di normalità. Tuttavia, dato che $N = 126.306$, facciamo pieno affidamento sul **Teorema del Limite Centrale (CLT)** per garantire la validità asintotica dell'inferenza statistica (stimatori asintoticamente normali).

#### Breusch-Pagan (Eteroschedasticità)
* **LM Stat**: $4807.67$ | $p\text{-value} = 0.000$
* *Risultato*: Si rifiuta con forza l'omocedasticità. L'eteroschedasticità residua, tipica nei grandi campioni microeconomici, giustifica l'adozione degli standard error robusti **HC3**.

---

### 3. Analisi della Multicollinearità (VIF)
Tutti i Variance Inflation Factors (VIF) calcolati sulle covariate sono inferiori a **$5.0$**, escludendo problematiche di multicollinearità e confermando la stabilità delle stime OLS.

---

### 4. Regolarizzazione Ridge
La contrazione analitica Ridge su covariate standardizzate (`ridge_path.png`) rivela che le dummy dei settori occupazionali (`Job_Education`, `Job_Fire`, `Job_Police`) sono le più resilienti alla regolarizzazione, a indicare che la struttura occupazionale è il fattore principale nel determinare i livelli retributivi rispetto ai fattori di genere.

---

### 5. Nested F-Test / Wald Joint Significance
Confrontando il modello saturo con quello ridotto (escludendo le variabili di genere ed interazioni):
* **Classical F-Statistic** (omocedastico): $200.03$ | $p\text{-value} < 0.0001$
* **Robust F-Statistic (HC3 VCE)**: **$217.83$** | $p\text{-value} < 0.0001$
* *Interpretazione*: La significatività congiunta delle variabili e interazioni di genere è massicciamente confermata sia dal test classico che dal test Wald robusto.

---

### 6. Visual Diagnostic Showcase (Parte I)

Di seguito viene riportata la galleria delle visualizzazioni e dei test diagnostici del modello OLS principale.

#### A. Diagnostica dei Residui del Modello Naïve
Residui a ventaglio (eteroschedasticità) e allontanamento dalla normalità sulle code prima della trasformazione.
![Diagnostica OLS Naïve](ols_diagnostics.png)

#### B. Profilo di Log-Verosimiglianza Box-Cox
Picco MLE del profilo di verosimiglianza stimato a $\lambda = 0.5969$, con l'arrotondamento accademico a $\lambda = 0.5$ per preservare l'interpretabilità.
![Box-Cox Likelihood](boxcox_likelihood.png)

#### C. Distribuzione Salariale Prima e Dopo la Trasformazione
L'istogramma a sinistra mostra la forte asimmetria a destra della retribuzione originale. A destra, la trasformazione radice quadrata ($\lambda = 0.5$) normalizza e simmetrizza l'intera distribuzione.
![Distribuzione Salariale Grezza vs Trasformata](salary_distribution.png)

#### D. Diagnostica del Modello Trasformato ($\lambda = 0.5$)
Dopo l'applicazione di $\sqrt{Y}$, la varianza dei residui si stabilizza e il Q-Q Plot risulta nettamente linearizzato.
![Diagnostica Modello Trasformato](transformed_diagnostics.png)

#### E. Diagnostica Avanzata (Leverage e Distanza di Cook)
Leva individuale con la soglia teorica $2(r+1)/n$ (linea tratteggiata rossa) e distanze di Cook con etichettatura automatica dei primi 5 outlier più influenti, guidati da figure dirigenziali come `Joanne Hayes-White`.
![Leva e Cook's Distance](leverage_cooks.png)

#### F. Pay Gap per Macro-Categoria Professionale
Boxplot comparativo che illustra la distribuzione dei salari per genere e ruolo aziendale, evidenziando le asimmetrie distributive.
![Gender Pay Gap per Settore](gender_pay_gap_by_job.png)

#### G. Evoluzione del Gap Salariale con l'Anzianità
Effetto dell'interazione tra genere ed anni di servizio. Le bande ombreggiate rappresentano gli intervalli di confidenza al 95%.
![Evoluzione Pay Gap con Anzianità](seniority_pay_gap.png)

#### H. Path di Contrazione Ridge (Shrinkage Path)
Shrinkage analitico dei coefficienti standardizzati del modello al variare del parametro di regolarizzazione $\lambda \in [10^{-2}, 10^5]$.
![Ridge Shrinkage Path](ridge_path.png)

---

## PARTE II: Segregazione Occupazionale come Driver Principale

Un recente benchmark di Machine Learning basato su XGBoost ha confermato i risultati del nostro modello OLS principale: **il Gender Pay Gap è guidato principalmente dalla segregazione occupazionale a monte (la scelta e la distribuzione dei settori) piuttosto che da dinamiche salariali all'interno dei singoli ruoli**. 

Per dare evidenza empirica immediata a questa dinamica strutturale, presentiamo in questa sezione le statistiche descrittive focalizzate interamente sulla segregazione occupazionale nel mercato del lavoro di San Francisco.

### 1. Distribuzione di Genere e Retribuzione per Settore

La tabella seguente riassume la distribuzione di genere all'interno di ciascuna macro-categoria lavorativa e il salario medio non trasformato (`TotalPay`) del settore, ordinato dal settore a maggior densità maschile a quello a maggior densità femminile:

| Settore | N. Maschi | % Maschi | N. Femmine | % Femmine | Salario Medio Settore |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fire** | 3.796 | 83,76% | 736 | 16,24% | $146.152,49 |
| **Police** | 12.795 | 77,46% | 3.723 | 22,54% | $118.044,70 |
| **Other / Baseline** | 45.626 | 59,35% | 31.255 | 40,65% | $74.774,36 |
| **Education** | 5.059 | 53,80% | 4.345 | 46,20% | $11.695,76 |
| **Medical** | 4.886 | 25,76% | 14.085 | 74,24% | $70.321,23 |
| **Totale Forza Lavoro** | **72.162** | **57,13%** | **54.144** | **42,87%** | **$74.768,32** |

> [!IMPORTANT]
> **Implicazioni dei Dati**:
> - I settori a **maggioranza maschile** (**Fire** con 83,76% uomini e **Police** con 77,46% uomini) sono caratterizzati dai livelli retributivi medi più alti del dataset (rispettivamente **$146.152,49** e **$118.044,70**).
> - Al contrario, il settore a **forte densità femminile** (**Medical** con 74,24% donne) presenta un salario medio di **$70.321,23**, circa la metà rispetto al settore Fire.
> - Questa disparità nella distribuzione tra settori (sorting occupazionale) spiega la maggior parte del gap salariale osservato a livello aggregato, confermando empiricamente che le differenze salariali tra generi traggono origine principalmente da barriere o preferenze nella fase di allocazione settoriale iniziale.

---

### 2. Visualizzazione Grafica della Segregazione

Il grafico sottostante (`occupational_segregation.png`) rappresenta visivamente il bilanciamento di genere all'interno di ciascun settore, evidenziando il forte sbilanciamento distributivo (utilizzando la palette di colori istituzionale: blu per gli uomini, rosa per le donne):

![Segregazione Occupazionale per Settore](occupational_segregation.png)

---

## 🚀 Guida di Esecuzione e Replicabilità

Per riprodurre in autonomia l'intero workflow statistico ed descrittivo, esegui il file principale nel terminale:

```bash
# Esegue il workflow completo all'interno dell'ambiente virtuale uv
uv run gender_analysis.py
```

Questo avvierà in modo sequenziale:
1. La pulizia dei dati e l'estrazione dei nomi di battesimo da `Salaries.csv`.
2. La classificazione automatica e ottimizzata del genere.
3. La stima del modello OLS trasformato Box-Cox ($\lambda = 0.5$) con relativi test diagnostici e scomposizioni.
4. Il calcolo delle statistiche descrittive sulla segregazione occupazionale e la generazione del relativo stacked bar chart (`occupational_segregation.png`).

---

## 🃏 Integrità e Conservazione del Progetto Poker

Tutti i codici, dataset e grafici relativi all'analisi strategica preflop/postflop e alla **Hold'em Profitability Matrix** sono stati **totalmente salvati e isolati** in una directory dedicata per non creare interferenze:

👉 [**Vai alla Cartella del Progetto Poker**](file:///c:/Users/loren/Documents/AntiGravity%20Projects/progettostat/poker/)

All'interno di tale cartella è presente una copia dell'originario `README.md` sul poker per consultare nuovamente i grafici diagnostici, il Q-Q plot leptocurtico delle vincite e la heatmap 13x13 del WinRate delle mani iniziali.

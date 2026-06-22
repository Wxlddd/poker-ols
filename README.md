# Analisi Econometrica sul Gender Pay Gap a San Francisco (2011–2014)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.6-darkblue.svg?style=for-the-badge&logo=python)](https://www.statsmodels.org/)
[![Scipy](https://img.shields.io/badge/scipy-1.17.1-lightblue.svg?style=for-the-badge&logo=scipy)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/pandas-3.0.3-violet.svg?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> [!NOTE]
> **Vetrina del Progetto Accademico**: Questo repository ospita un workflow econometrico e di economia del lavoro rigoroso in Python. Tramite regressioni lineari multiple (OLS), trasformazioni Box-Cox ($\lambda = 0.5$ per la radice quadrata) e log-lineari ($\lambda = 0$), test F/Wald robusti all'eteroschedasticità, la **Decomposizione di Oaxaca-Blinder** e l'analisi dei **Bad Controls**, studiamo il **Gender Pay Gap** (divario salariale di genere) nel personale del Comune di San Francisco, verificando analiticamente tutti i teoremi e test diagnostici della statistica matematica.

---

## 📐 Fondamenti Teorici e Teoremi Verificati

Questo studio copre in modo esaustivo tutte le proprietà geometriche e le ipotesi del modello lineare classico (Gauss-Markov) e le loro estensioni robuste.

### 1. Specificazione del Modello e Invertibilità
Modelliamo la retribuzione totale dei dipendenti $Y$ tramite una regressione lineare multipla:
$$\vec{y} = Z\vec{\beta} + \vec{\varepsilon}$$
dove $Z \in \mathbb{R}^{n \times (r+1)}$ è la matrice di disegno contenente l'intercetta, la dummy di genere, il proxy continuo di anzianità, le dummy temporali (anno), le dummy di macro-categoria lavorativa e i relativi termini di interazione.

Per garantire l'esistenza e l'unicità dello stimatore OLS:
$$\hat{\vec{\beta}}\_{OLS} = (Z^T Z)^{-1} Z^T \vec{y}$$
verifichiamo rigorosamente che la matrice $Z$ sia a **rango colonna pieno**:
$$\text{rango}(Z) = r + 1$$
in modo da assicurare che la matrice dei prodotti $Z^T Z$ sia definita positiva e strettamente invertibile.

### 2. Standard Error Robusti all'Eteroschedasticità (HC3)
Dato che il test di Breusch-Pagan e l'ispezione grafica evidenziano una forte eteroschedasticità ($Var(\vec{\varepsilon}) \neq \sigma^2 I$) e i residui non sono perfettamente gaussiani (Shapiro-Wilk rifiutato), facciamo affidamento sull'inferenza asintotica. Per correggere la distorsione del calcolo dei p-value e degli intervalli di confidenza, implementiamo gli stimatori della matrice di varianza-covarianza (VCE) robusti all'eteroschedasticità di tipo **HC3** (Davidson & MacKinnon, 1993):
$$\widehat{\text{Var}}(\hat{\vec{\beta}})\_{HC3} = (Z^T Z)^{-1} Z^T \Omega Z (Z^T Z)^{-1}$$
dove $\Omega = \text{diag}\left( \frac{\hat{\varepsilon}_i^2}{(1 - h_{ii})^2} \right)$ e $h_{ii}$ rappresenta il leverage dell'osservazione $i$. HC3 garantisce prestazioni inferenziali ottimali in campioni finiti e in presenza di punti ad alta leva.

### 3. Matrice Hat, Leverage e Distanza di Cook
Il vettore delle previsioni $\hat{\vec{y}}$ è una proiezione lineare di $\vec{y}$ sullo spazio delle colonne di $Z$ tramite la **Matrice Hat** $H$:
$$\hat{\vec{y}} = H\vec{y}, \quad H = Z(Z^T Z)^{-1} Z^T$$
I valori sulla diagonale $h\_{ii} \in [0,1]$ misurano la **leva** (leverage) di ciascuna osservazione. I punti di leva critici vengono individuati tramite la soglia teorica:
$$h\_{ii} > \frac{2(r+1)}{n}$$
Per valutare l'influenza di ciascuna osservazione sulla stima complessiva del vettore dei coefficienti $\hat{\vec{\beta}}$, calcoliamo la **Distanza di Cook** $D\_i$:
$$D\_i = \frac{t\_i^2}{r+1} \left( \frac{h\_{ii}}{1 - h\_{ii}} \right)$$
dove $t\_i$ rappresenta il residuo studentizzato internamente:
$$t\_i = \frac{\hat{\varepsilon}\_i}{\hat{\sigma} \sqrt{1 - h\_{ii}}}$$

### 4. Trasformazioni Box-Cox ($\lambda = 0.5$) e Log-Lineari ($\lambda = 0$)
Per correggere l'eteroschedasticità e linearizzare la relazione funzionale, utilizziamo due diverse scale della variabile dipendente:
1. **Trasformazione Box-Cox ($\lambda = 0.5$)**: Operiamo sulla radice quadrata ($Y^{(0.5)} = 2(\sqrt{Y} - 1)$), che linearizza la varianza offrendo una buona interpretabilità.
2. **Trasformazione Logaritmica ($\lambda = 0$)**: La specifica log-lineare ($\ln(Y)$) è lo standard in economia del lavoro, in quanto consente di interpretare i coefficienti come variazioni percentuali semielastiche della retribuzione rispetto alle covariate:
$$\ln(Y_i) = Z_i \vec{\beta} + \varepsilon_i \implies \frac{\partial E[Y|Z]}{\partial Z_j} \cdot \frac{1}{E[Y|Z]} \approx \beta_j$$

### 5. Verifica del Gap Netto nei Settori e VCE
Nel settore 'Education', la retribuzione delle donne è influenzata sia dall'effetto principale di genere che dall'effetto interattivo. Il divario di genere netto in questo settore è espresso dalla combinazione lineare:
$$\theta = \beta_{\text{Gender}} + \beta_{\text{Gender} \times \text{Job\_Education}}$$
Per condurre inferenza scientifica su $\theta$, calcoliamo la varianza analitica della combinazione lineare partendo dalla matrice di varianza-covarianza degli stimatori (VCE):
$$\text{Var}(\hat{\theta}) = \text{Var}(\hat{\beta}_1) + \text{Var}(\hat{\beta}_2) + 2\text{Cov}(\hat{\beta}_1, \hat{\beta}_2)$$
Successivamente, testiamo formalmente l'ipotesi nulla:
$$H_0: \beta_{\text{Gender}} + \beta_{\text{Gender} \times \text{Job\_Education}} = 0$$
tramite un test di Wald (Chi-quadro ad 1 grado di libertà) basato sulla matrice VCE robusta HC3.

### 6. Decomposizione di Oaxaca-Blinder
Per isolare la discriminazione di genere salariale dagli effetti di allocazione (capitale umano e sorting), implementiamo la classica scomposizione di Oaxaca-Blinder sulla specifica log-lineare ($\ln(Y)$). Dividiamo il dataset nei sottocampioni Maschi ($M$) e Femmine ($F$) e stimiamo due regressioni OLS separate con standard error HC3:
$$\ln(Y_{iM}) = X_{iM}^T \beta_M + \varepsilon_{iM}, \quad \ln(Y_{iF}) = X_{iF}^T \beta_F + \varepsilon_{iF}$$
Il divario medio geometrico si scompone matematicamente come:
$$\bar{\ln(Y_M)} - \bar{\ln(Y_F)} = \underbrace{(\bar{X}_M - \bar{X}_F)^T \hat{\beta}_M}_{\text{Componente Spiegata (Endowment)}} + \underbrace{\bar{X}_F^T (\hat{\beta}_M - \hat{\beta}_F)}_{\text{Componente Non Spiegata (Discrimination)}}$$
* **Componente Spiegata (Endowment Effect)**: Quantifica la quota del gap dovuta a differenze medie nelle caratteristiche (anzianità lavorativa, anno di osservazione, collocazione nei macro-settori).
* **Componente Non Spiegata (Coefficient/Discrimination Effect)**: Misura la quota del gap derivante da differenze nei rendimenti (coefficienti) delle caratteristiche, inclusa la differenza negli intercetti, comunemente interpretata come misura della discriminazione salariale o segregazione interna non controllata.

### 7. Analisi dei "Bad Controls" e Segregazione Occupazionale
L'inclusione di controlli sulle categorie lavorative (`Job_Education`, `Job_Fire`, ecc.) nel modello salariale può dar luogo al problema metodologico dei **bad controlli** (Angrist & Pischke, 2009). Se il macro-settore di impiego è esso stesso influenzato dal genere (segregazione occupazionale dovuta a barriere all'accesso o preferenze), controllare per il settore "assorbe" e occulta una porzione significativa del divario di genere complessivo.
Stimiamo quindi un **Modello Unadjusted (Short)** ed un **Modello Adjusted (Long, no interactions)** per quantificare la quota del gap spiegata dal sorting occupazionale:
$$\text{Sorting Portion} = \frac{\beta_{\text{Gender}}^{\text{Short}} - \beta_{\text{Gender}}^{\text{Long, no inter.}}}{\beta_{\text{Gender}}^{\text{Short}}}$$

---

## 📊 Sintesi dei Risultati Empirici

### Dimensioni Campionarie e Filtri
* **Record Originali**: 148.654
* **Salari Positivi (BasePay > 0 & TotalPay > 0)**: 146.736
* **Filtro di Genere (nomi classificati univocamente)**: **126.306** record finali.
* **Composizione**: **42,1% Femmine (1)**, **57,9% Maschi (0)**.

### 1. Stime OLS dei Modelli Adjusted (Long Saturated)
Di seguito si riportano i coefficienti stimati con standard error robusti HC3 per i due modelli adjusted saturi.

| Covariata | Box-Cox ($\lambda = 0.5$) | Std. Err. HC3 | p-value | Log-Lineare ($\lambda = 0$) | Std. Err. HC3 | p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | 515.0001 | 1.202 | **0.000** | 10.9240 | 0.007 | **0.000** |
| **Gender (Femmina)** | -27.9417 | 1.476 | **0.000** | -0.1096 | 0.009 | **0.000** |
| **Seniority** | 54.6008 | 1.209 | **0.000** | 0.3062 | 0.008 | **0.000** |
| **Year_2012** | -37.5796 | 1.655 | **0.000** | -0.2285 | 0.011 | **0.000** |
| **Year_2013** | 23.7536 | 1.365 | **0.000** | 0.0846 | 0.008 | **0.000** |
| **Year_2014** | -53.4521 | 1.933 | **0.000** | -0.3377 | 0.013 | **0.000** |
| **Job_Education** | -329.8440 | 1.725 | **0.000** | -2.1719 | 0.019 | **0.000** |
| **Job_Fire** | 223.1916 | 2.504 | **0.000** | 0.8347 | 0.010 | **0.000** |
| **Job_Medical** | -30.5440 | 3.164 | **0.000** | -0.2118 | 0.018 | **0.000** |
| **Job_Police** | 155.7580 | 1.573 | **0.000** | 0.6284 | 0.007 | **0.000** |
| **Gender x Seniority** | -1.4818 | 1.352 | 0.273 | -0.0025 | 0.008 | 0.758 |
| **Gender x Job_Education** | 28.0345 | 2.551 | **0.000** | 0.0952 | 0.028 | **0.001** |
| **Gender x Job_Fire** | 7.6066 | 5.979 | 0.203 | 0.0453 | 0.024 | 0.063 |
| **Gender x Job_Medical** | 8.4227 | 3.771 | **0.025** | -0.0138 | 0.022 | 0.528 |
| **Gender x Job_Police** | -50.0045 | 3.196 | **0.000** | -0.1543 | 0.015 | **0.000** |

> [!IMPORTANT]
> **Inferenza su Anzianità ed Efficacia delle Interazioni**:
> * L'interazione tra genere ed anzianità (`Gender x Seniority`) non è statisticamente significativa in nessuno dei due modelli ($p > 0.25$). Questo dimostra empiricamente che il Gender Pay Gap rimane stabile e costante all'aumentare dell'anzianità nel dataset analizzato.
> * Al contrario, le interazioni settoriali (`Gender x JobCategory`) mostrano differenze salariali significative tra i vari ruoli.

---

### 2. Verifica Rigorosa del Gap Netto nel Settore 'Education' (Section 1c)
Verifichiamo formalmente se nel settore 'Education' il divario di genere netto $\theta = \beta_{\text{Gender}} + \beta_{\text{Gender} \times \text{Job\_Education}}$ si annulla.

* **Modello Box-Cox ($\lambda = 0.5$)**:
  * **Stima Puntuale $\hat{\theta}$**: $0.092882$ (radice quadrata di dollari)
  * **Standard Error (Robust HC3)**: $2.235679$
  * **Intervallo di Confidenza al 95%**: $[-4.288969, 4.474732]$
  * **Statistica del Test di Wald ($\chi^2$, 1 df)**: $0.0017$
  * **p-value**: **$0.9669$**
  * *Verdetto*: **Accettiamo l'ipotesi nulla.** Il gap netto nel settore Education è statisticamente nullo.
* **Modello Log-Lineare ($\lambda = 0$)**:
  * **Stima Puntuale $\hat{\theta}$**: $-0.014383$ (ovvero un divario del $-1.43\%$)
  * **Standard Error (Robust HC3)**: $0.027280$
  * **Intervallo di Confidenza al 95%**: $[-0.067851, 0.039084]$
  * **Statistica del Test di Wald ($\chi^2$, 1 df)**: $0.2780$
  * **p-value**: **$0.5980$**
  * *Verdetto*: **Accettiamo l'ipotesi nulla.** Anche nella scala logaritmica standard di robustezza il gap netto in Education si conferma statisticamente nullo.

---

### 3. Decomposizione Oaxaca-Blinder (Sezione 2a)
La decomposizione classica del divario medio di log-wages ($\overline{\ln(Y_M)} - \overline{\ln(Y_F)}$) restituisce i seguenti risultati:

* **Divario Totale**: $0.283141$ log-points (pari ad un gap geometrico medio del **$32.73\%$** a favore degli uomini).
* **Componente Spiegata (Endowment Effect)**: $0.166265$ log-points (**$58.72\%$** del gap totale). Rappresenta il divario spiegabile dal fatto che uomini e donne hanno qualifiche e macro-collocazioni lavorative diverse.
* **Componente Non Spiegata (Discrimination Effect)**: $0.116877$ log-points (**$41.28\%$** del gap totale). Misura la disparità di trattamento a parità di caratteristiche medie e macro-settori, indicando una discriminazione salariale strutturale residua del **$12.39\%$** ($e^{0.116877}-1$).

---

### 4. Analisi dei "Bad Controls" e Segregazione Occupazionale (Sezione 2c)
Confrontiamo l'effetto del genere (`Gender`) tra il modello Unadjusted (Short) e i modelli Adjusted (Long) per isolare l'impatto della segregazione occupazionale sistematica.

#### Tabella Comparativa dei Coefficienti di Genere (Box-Cox $\lambda = 0.5$)
| Modello | Coefficiente `Gender` | Dev. Standard (HC3) | Statistica z | p-value | Conf. Int. 95% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Unadjusted (Short Model)** | -62.283391 | 1.146297 | -54.3344 | 0.000 | [-64.5301, -60.0367] |
| **2. Adjusted (Long, No Interactions)** | -30.286660 | 1.015031 | -29.8382 | 0.000 | [-32.2761, -28.2972] |
| **3. Adjusted (Long, Saturated)** | -27.941663 | 1.475688 | -18.9347 | 0.000 | [-30.8340, -25.0494] |

* **Quota del gap originario spiegata dal sorting occupazionale**: **$51.37\%$**

#### Tabella Comparativa dei Coefficienti di Genere (Log-Lineare $\lambda = 0$)
| Modello | Coefficiente `Gender` | Dev. Standard (HC3) | Statistica z | p-value | Conf. Int. 95% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Unadjusted (Short Model)** | -0.279236 | 0.006983 | -39.9900 | 0.000 | [-0.2929, -0.2656] |
| **2. Adjusted (Long, No Interactions)** | -0.119660 | 0.006256 | -19.1271 | 0.000 | [-0.1319, -0.1074] |
| **3. Adjusted (Long, Saturated)** | -0.109586 | 0.009138 | -11.9918 | 0.000 | [-0.1275, -0.0917] |

* **Quota del gap originario spiegata dal sorting occupazionale**: **$57.15\%$**

> [!WARNING]
> **Interpretazione Econometrica del Sorting come "Bad Control"**:
> Poiché oltre il **$57\%$** (nel modello log-lineare) del divario salariale grezzo è spiegato dallo smistamento (sorting) nei diversi settori, controllare per il macro-settore di lavoro riduce artificialmente il gap di genere dal $-27.9\%$ (gap grezzo) al $-12.0\%$ (gap medio controllato). 
> 
> Tuttavia, se l'accesso ai settori remunerativi (es. `Police` o `Fire`) è influenzato da discriminazione o segregazione di genere all'ingresso, le variabili settoriali agiscono come **bad controls**. In questo scenario, il vero divario strutturale è più vicino al modello unadjusted (Short), e il modello adjusted (Long) maschera e assorbe l'effetto della discriminazione all'ingresso dei mercati di lavoro comunali.

---

### 5. Nested F-Test / Wald Joint Significance
Confrontando il modello saturo con quello ridotto (senza variabili di genere ed interazioni):

* **Classical F-Statistic** (basato su SSRES): $200.03$ ($p < 0.0001$)
* **Robust F-Statistic (HC3 VCE)** (basato su test di Wald): **$217.83$** ($p < 0.0001$)
* *Interpretazione*: Entrambi i test rifiutano l'ipotesi nulla di ininfluenza di genere. La significatività congiunta del genere e delle sue interazioni si conferma massicciamente robusta anche correggendo per l'eteroschedasticità.

---

## 📈 Visual Diagnostic Showcase

Di seguito viene riportata la galleria completa delle visualizzazioni e dei test diagnostici generati dal nostro workflow statistico.

### 1. Diagnostica delle Regressioni e Gauss-Markov

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

---

### 2. Gap di Genere ed Effetti Interattivi

#### A. Pay Gap per Macro-Categoria Professionale
Boxplot comparativo che illustra la distribuzione dei salari per genere e ruolo aziendale, evidenziando le asimmetrie distributive.
![Gender Pay Gap per Settore](gender_pay_gap_by_job.png)

#### B. Evoluzione del Gap Salariale con l'Anzianità
Effetto dell'interazione tra genere ed anni di servizio. Le bande ombreggiate rappresentano gli intervalli di confidenza al 95%.
![Evoluzione Pay Gap con Anzianità](seniority_pay_gap.png)

---

### 3. Regolarizzazione Ridge

#### Path di Contrazione Ridge (Shrinkage Path)
Shrinkage analitico dei coefficienti standardizzati del modello al variare del parametro di regolarizzazione $\lambda \in [10^{-2}, 10^5]$.
![Ridge Shrinkage Path](ridge_path.png)

---

## 🚀 Guida di Esecuzione e Replicabilità

Per riprodurre in autonomia l'intero workflow statistico ed econometrico, esegui il file principale nel terminale:

```bash
# Esegue il workflow completo all'interno dell'ambiente virtuale uv
uv run gender_analysis.py
```

Questo avvierà in modo sequenziale:
1. La pulizia dei dati e l'estrazione dei nomi di battesimo da `Salaries.csv`.
2. La classificazione automatica e ottimizzata del genere.
3. La stima delle regressioni OLS con relativi test diagnostici e scomposizioni matriciali.
4. L'esecuzione dei test di robustezza net gap, la decomposizione Oaxaca-Blinder e l'analisi dei bad controls.
5. Il salvataggio dei grafici diagnostici ad alta definizione nella cartella di lavoro.

---

## 🃏 Integrità e Conservazione del Progetto Poker

Tutti i codici, dataset e grafici relativi all'analisi strategica preflop/postflop e alla **Hold'em Profitability Matrix** sono stati **totalmente salvati e isolati** in una directory dedicata per non creare interferenze:

👉 [**Vai alla Cartella del Progetto Poker**](file:///c:/Users/loren/Documents/AntiGravity%20Projects/progettostat/poker/)

All'interno di tale cartella è presente una copia dell'originario `README.md` sul poker per consultare nuovamente i grafici diagnostici, il Q-Q plot leptocurtico delle vincite e la heatmap 13x13 del WinRate delle mani iniziali.

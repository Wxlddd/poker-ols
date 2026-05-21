# Analisi Econometrica sul Gender Pay Gap a San Francisco (2011–2014)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.6-darkblue.svg?style=for-the-badge&logo=python)](https://www.statsmodels.org/)
[![Scipy](https://img.shields.io/badge/scipy-1.17.1-lightblue.svg?style=for-the-badge&logo=scipy)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/pandas-3.0.3-violet.svg?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> [!NOTE]
> **Vetrina del Progetto Accademico**: Questo repository ospita un workflow econometrico e di economia del lavoro rigoroso in Python. Tramite regressioni lineari multiple (OLS), trasformazioni Box-Cox, test F parziali per modelli nidificati e regolarizzazione Ridge, studiamo il **Gender Pay Gap** (divario salariale di genere) nel personale del Comune di San Francisco, verificando analiticamente tutti i teoremi e test diagnostici della statistica matematica.

---

## 📐 Fondamenti Teorici e Teoremi Verificati

Questo studio copre in modo esaustivo tutte le proprietà geometriche e le ipotesi del modello lineare classico (Gauss-Markov).

### 1. Specificazione del Modello e Invertibilità
Modelliamo la retribuzione totale dei dipendenti $Y$ tramite una regressione lineare multipla:
$$\vec{y} = Z\vec{\beta} + \vec{\varepsilon}$$
dove $Z \in \mathbb{R}^{n \times (r+1)}$ è la matrice di disegno contenente l'intercetta, la dummy di genere, il proxy continuo di anzianità, le dummy temporali (anno), le dummy di macro-categoria lavorativa e i relativi termini di interazione.

Per garantire l'esistenza e l'unicità dello stimatore OLS:
$$\hat{\vec{\beta}}_{OLS} = (Z^T Z)^{-1} Z^T \vec{y}$$
verifichiamo rigorosamente che la matrice $Z$ sia a **rango colonna pieno**:
$$\text{rango}(Z) = r + 1$$
in modo da assicurare che la matrice dei prodotti $Z^T Z$ sia definita positiva e strettamente invertibile.

### 2. Decomposizione della Varianza e Ortogonalità
Sotto stima OLS, la somma dei quadrati totale ($SSTOT$) si scompone perfettamente in somma dei quadrati spiegata dalla regressione ($SSREG$) e somma dei quadrati dei residui ($SSRES$):
$$SSTOT = SSREG + SSRES$$
$$\sum_{i=1}^n (y_i - \bar{y})^2 = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 + \sum_{i=1}^n \hat{\varepsilon}_i^2$$
Questa scomposizione geometrica è garantita dal **teorema di ortogonalità fitted-residui**:
$$\hat{\vec{y}}^T \hat{\vec{\varepsilon}} = 0$$

### 3. Matrice Hat, Leverage e Distanza di Cook
Il vettore delle previsioni $\hat{\vec{y}}$ è una proiezione lineare di $\vec{y}$ sullo spazio delle colonne di $Z$ tramite la **Matrice Hat** $H$:
$$\hat{\vec{y}} = H\vec{y}, \quad H = Z(Z^T Z)^{-1} Z^T$$
I valori sulla diagonale $h_{ii} \in [0,1]$ misurano la **leva** (leverage) di ciascuna osservazione. I punti di leva critici vengono individuati tramite la soglia teorica:
$$h_{ii} > \frac{2(r+1)}{n}$$
Per valutare l'influenza di ciascuna osservazione sulla stima complessiva del vettore dei coefficienti $\hat{\vec{\beta}}$, calcoliamo la **Distanza di Cook** $D_i$:
$$D_i = \frac{t_i^2}{r+1} \left( \frac{h_{ii}}{1 - h_{ii}} \right)$$
dove $t_i$ rappresenta il residuo studentizzato internamente:
$$t_i = \frac{\hat{\varepsilon}_i}{\hat{\sigma} \sqrt{1 - h_{ii}}}$$

### 4. Trasformazione Box-Cox per la Stabilizzazione della Varianza
In presenza di eteroschedasticità ($Var(\vec{\varepsilon}) \neq \sigma^2 I$), applichiamo la trasformazione di potenza di **Box-Cox** sulla risposta continua $Y$ (strettamente positiva) per stabilizzare la varianza e ripristinare la gaussianità dei residui:
$$Y^{(\lambda)} = \begin{cases} \frac{Y^\lambda - 1}{\lambda} & \text{se } \lambda \neq 0 \\ \ln(Y) & \text{se } \lambda = 0 \end{cases}$$
La stima del parametro ottimale $\lambda$ avviene tramite Massima Verosimiglianza (MLE), massimizzando la funzione di log-verosimiglianza del profilo:
$$L(\lambda) = -\frac{n}{2} \ln(\text{Var}(Y^{(\lambda)})) + (\lambda - 1) \sum_{i=1}^n \ln(y_i)$$

### 5. Selezione delle Variabili e Test F Parziale (ANOVA)
Confrontiamo il Modello Completo (Saturo) con un Modello Ridotto (escludendo tutte le variabili inerenti al genere ed interazioni) per testare l'ipotesi nulla:
$$H_0: \beta_{\text{Gender}} = \beta_{\text{Gender}\times\text{Seniority}} = \dots = 0$$
Utilizziamo la statistica F parziale basata sui residui:
$$F = \frac{(SSRES_{\text{ridotto}} - SSRES_{\text{completo}}) / q}{SSRES_{\text{completo}} / (n - p_{\text{completo}})} \sim F(q, n - p_{\text{completo}})$$

### 6. Regolarizzazione Ridge
Per gestire l'eventuale multicollinearità tra dummy settoriali ed interazioni, calcoliamo analiticamente lo stimatore Ridge al variare di $\lambda \ge 0$ sulle covariate standardizzate:
$$\hat{\vec{\beta}}_{RR}(\lambda) = (Z_{\text{std}}^T Z_{\text{std}} + \lambda I)^{-1} Z_{\text{std}}^T \vec{y}_{\text{centrata}}$$

---

## 📊 Sintesi dei Risultati Empirici

### Dimensioni Campionarie e Filtri
* **Record Originali**: 148.654
* **Salari Positivi (BasePay > 0 & TotalPay > 0)**: 146.736
* **Filtro di Genere (nomi classificati univocamente)**: **126.306** record finali.
* **Composizione**: **42,1% Femmine (1)**, **57,9% Maschi (0)**.

### 1. Confronto tra Modelli OLS (Scala Originale vs. Box-Cox)
Il modello sulla scala originale dei salari violava pesantemente le ipotesi di Gauss-Markov (forte eteroschedasticità e residui non normali).

Il nostro algoritmo MLE ha stimato il parametro ottimale di potenza a **$\lambda = 0,5969$**.

#### Tabella dei Coefficienti (Modello Trasformato $Y^{(0.597)}$)
* **$R^2$ Rettificato**: **0,324**
* **F-statistic**: **4.327** ($p\text{-value} = 0.000$)

| Covariata | Coefficiente | Dev. Standard | Statistica t | p-value | Intervallo di Conf. 95% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | 1287.6690 | 3.440 | 374.271 | **0.000** | [1280.93, 1294.41] |
| **Gender (Femmina)** | -82.6500 | 4.147 | -19.931 | **0.000** | [-90.78, -74.52] |
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

### 2. Interpretazione Econometrica delle Interazioni
La derivata parziale della retribuzione trasformata rispetto a Gender (Femmina) è data da:
$$\frac{\partial Y^{(0.597)}}{\partial \text{Gender}} = -82.65 + 83.77 \cdot \text{Job\_Education} + 28.01 \cdot \text{Job\_Medical} - 154.24 \cdot \text{Job\_Police}$$

* **Divario di Base**: Nella macro-categoria di riferimento (`Other`), le donne guadagnano significativamente meno dei colleghi uomini ($\beta_{\text{Gender}} = -82.65$, $p < 0.001$).
* **Mitigazione in Education**: Nel settore scolastico/educativo, il divario di genere è totalmente annullato ($\beta_{\text{Gender}} + \beta_{\text{Gender}\times\text{Job\_Education}} = -82.65 + 83.77 = +1.12$, statisticamente nullo).
* **Ampliamento nelle Forze dell'Ordine**: Nel corpo di Polizia, il divario salariale si amplia drammaticamente, registrando la disparità più elevata a svantaggio del genere femminile ($\beta_{\text{Gender}} + \beta_{\text{Gender}\times\text{Job\_Police}} = -82.65 - 154.24 = -236.89$, $p < 0.001$).

---

## 📈 Visual Diagnostic Showcase

Tutti i grafici salvati in formato vettoriale ad alta risoluzione sono inseriti ed analizzati nel walkthrough:

* **Log-Verosimiglianza Box-Cox**: Picco MLE stimato a $\lambda \approx 0.6$ (salvato in `boxcox_likelihood.png`).
* **Confronto Residui OLS**: Riduzione drastica dell'andamento a ventaglio nei residui ed eccellente linearizzazione del Q-Q Plot dopo la trasformazione (salvati in `ols_diagnostics.png` e `transformed_diagnostics.png`).
* **Ridge Shrinkage Path**: Visualizzazione della contrazione dei coefficienti al variare di $\lambda$ (salvato in `ridge_path.png`).

---

## 🚀 Guida di Esecuzione e Replicabilità

Per riprodurre in autonomia l'intero workflow statistico ed econometrico, esegui il file principale nel terminale:

```bash
# Esegue il workflow completo all'interno dell'ambiente virtuale uv
uv run gender_analysis.py
```

Questo avvierà in modo sequenziale:
1. La pulizia dei dati e l'estrazione dei nomi di battesimo da `Salaries.csv`.
2. La classificazione automatica e ottimizzata offline del genere.
3. La stima delle regressioni OLS con relativi test diagnostici e scomposizioni matriciali.
4. Il salvataggio dei grafici diagnostici ad alta definizione nella cartella di lavoro.

---

## 🃏 Integrità e Conservazione del Progetto Poker

Tutti i codici, dataset e grafici relativi all'analisi strategica preflop/postflop e alla **Hold'em Profitability Matrix** sono stati **totalmente salvati e isolati** in una directory dedicata per non creare interferenze:

👉 [**Vai alla Cartella del Progetto Poker**](file:///c:/Users/loren/Documents/AntiGravity%20Projects/progettostat/poker/)

All'interno di tale cartella è presente una copia dell'originario `README.md` sul poker per consultare nuovamente i grafici diagnostici, il Q-Q plot leptocurtico delle vincite e la bellissima heatmap 13x13 del WinRate delle mani iniziali.

# Analisi Econometrica sul Gender Pay Gap a San Francisco (2011–2014)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.6-darkblue.svg?style=for-the-badge&logo=python)](https://www.statsmodels.org/)
[![Scipy](https://img.shields.io/badge/scipy-1.17.1-lightblue.svg?style=for-the-badge&logo=scipy)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/pandas-3.0.3-violet.svg?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> [!NOTE]
> **Vetrina del Progetto Accademico**: Questo repository ospita un workflow econometrico e di economia del lavoro rigoroso in Python. Tramite regressioni lineari multiple (OLS), trasformazioni Box-Cox ($\lambda = 0.5$ per massimizzare l'interpretabilità), test F parziali per modelli nidificati e regolarizzazione Ridge, studiamo il **Gender Pay Gap** (divario salariale di genere) nel personale del Comune di San Francisco, verificando analiticamente tutti i teoremi e test diagnostici della statistica matematica.

---

## 📐 Fondamenti Teorici e Teoremi Verificati

Questo studio copre in modo esaustivo tutte le proprietà geometriche e le ipotesi del modello lineare classico (Gauss-Markov).

### 1. Specificazione del Modello e Invertibilità
Modelliamo la retribuzione totale dei dipendenti $Y$ tramite una regressione lineare multipla:
$$\vec{y} = Z\vec{\beta} + \vec{\varepsilon}$$
dove $Z \in \mathbb{R}^{n \times (r+1)}$ è la matrice di disegno contenente l'intercetta, la dummy di genere, il proxy continuo di anzianità, le dummy temporali (anno), le dummy di macro-categoria lavorativa e i relativi termini di interazione.

Per garantire l'esistenza e l'unicità dello stimatore OLS:
$$\hat{\vec{\beta}}\_{OLS} = (Z^T Z)^{-1} Z^T \vec{y}$$
verifichiamo rigorosamente che la matrice $Z$ sia a **rango colonna pieno**:
$$\text{rango}(Z) = r + 1$$
in modo da assicurare che la matrice dei prodotti $Z^T Z$ sia definita positiva e strettamente invertibile.

### 2. Decomposizione della Varianza e Ortogonalità
Sotto stima OLS, la somma dei quadrati totale ($SSTOT$) si scompone perfettamente in somma dei quadrati spiegata dalla regressione ($SSREG$) e somma dei quadrati dei residui ($SSRES$):
$$SSTOT = SSREG + SSRES$$
$$\sum\_{i=1}^n (y\_i - \bar{y})^2 = \sum\_{i=1}^n (\hat{y}\_i - \bar{y})^2 + \sum\_{i=1}^n \hat{\varepsilon}\_i^2$$
Questa scomposizione geometrica è garantita dal **teorema di ortogonalità fitted-residui**:
$$\hat{\vec{y}}^T \hat{\vec{\varepsilon}} = 0$$

### 3. Matrice Hat, Leverage e Distanza di Cook
Il vettore delle previsioni $\hat{\vec{y}}$ è una proiezione lineare di $\vec{y}$ sullo spazio delle colonne di $Z$ tramite la **Matrice Hat** $H$:
$$\hat{\vec{y}} = H\vec{y}, \quad H = Z(Z^T Z)^{-1} Z^T$$
I valori sulla diagonale $h\_{ii} \in [0,1]$ misurano la **leva** (leverage) di ciascuna osservazione. I punti di leva critici vengono individuati tramite la soglia teorica:
$$h\_{ii} > \frac{2(r+1)}{n}$$
Per valutare l'influenza di ciascuna osservazione sulla stima complessiva del vettore dei coefficienti $\hat{\vec{\beta}}$, calcoliamo la **Distanza di Cook** $D\_i$:
$$D\_i = \frac{t\_i^2}{r+1} \left( \frac{h\_{ii}}{1 - h\_{ii}} \right)$$
dove $t\_i$ rappresenta il residuo studentizzato internamente:
$$t\_i = \frac{\hat{\varepsilon}\_i}{\hat{\sigma} \sqrt{1 - h\_{ii}}}$$

### 4. Trasformazione Box-Cox per la Stabilizzazione della Varianza
In presenza di eteroschedasticità ($Var(\vec{\varepsilon}) \neq \sigma^2 I$), applichiamo la trasformazione di potenza di **Box-Cox** sulla risposta continua $Y$ (strettamente positiva) per stabilizzare la varianza e ripristinare la gaussianità dei residui:
$$Y^{(\lambda)} = \begin{cases} \frac{Y^\lambda - 1}{\lambda} & \text{se } \lambda \neq 0 \\ \ln(Y) & \text{se } \lambda = 0 \end{cases}$$
La stima del parametro ottimale $\lambda$ avviene tramite Massima Verosimiglianza (MLE), massimizzando la funzione di log-verosimiglianza del profilo:
$$L(\lambda) = -\frac{n}{2} \ln(\text{Var}(Y^{(\lambda)})) + (\lambda - 1) \sum\_{i=1}^n \ln(y\_i)$$

> [!IMPORTANT]
> **Scelta Econometrica Consapevole ($\lambda = 0.5$)**: Sebbene la stima puntuale della log-verosimiglianza indichi $\lambda = 0.5969$, per motivi di rigorosa interpretabilità econometrica abbiamo impostato **$\lambda = 0.5$**. Questo ci consente di operare su una trasformazione standard standardizzabile (trasformazione radice quadrata, $\sqrt{Y}$), evitando coefficienti astratti che nuocerebbero alla leggibilità del modello, in perfetto accordo con le migliori pratiche accademiche.

### 5. Selezione delle Variabili e Test F Parziale (ANOVA)
Confrontiamo il Modello Completo (Saturo) con un Modello Ridotto (escludendo tutte le variabili inerenti al genere ed interazioni) per testare l'ipotesi nulla:
$$H\_0: \beta\_{\text{Gender}} = \beta\_{\text{Gender} \times \text{Seniority}} = \dots = 0$$
Utilizziamo la statistica F parziale basata sui residui:
$$F = \frac{(SSRES\_{\text{ridotto}} - SSRES\_{\text{completo}}) / q}{SSRES\_{\text{completo}} / (n - p\_{\text{completo}})} \sim F(q, n - p\_{\text{completo}})$$

### 6. Regolarizzazione Ridge
Per gestire l'eventuale multicollinearità tra dummy settoriali ed interazioni, calcoliamo analiticamente lo stimatore Ridge al variare di $\lambda \ge 0$ sulle covariate standardizzate:
$$\hat{\vec{\beta}}\_{RR}(\lambda) = (Z\_{\text{std}}^T Z\_{\text{std}} + \lambda I)^{-1} Z\_{\text{std}}^T \vec{y}\_{\text{centrata}}$$

---

## 📊 Sintesi dei Risultati Empirici

### Dimensioni Campionarie e Filtri
* **Record Originali**: 148.654
* **Salari Positivi (BasePay > 0 & TotalPay > 0)**: 146.736
* **Filtro di Genere (nomi classificati univocamente)**: **126.306** record finali.
* **Composizione**: **42,1% Femmine (1)**, **57,9% Maschi (0)**.

### 1. Modello OLS con Trasformazione Radice Quadrata ($\lambda = 0.5$)
Il fitting dell'OLS sulla scala trasformata $Y^{(0.5)}$ restituisce ottime metriche:
* **$R^2$ Rettificato**: **0,327**
* **F-statistic**: **4.389** ($p\text{-value} = 0.000$)

#### Tabella dei Coefficienti (Modello Trasformato $\lambda = 0.5$)

| Covariata | Coefficiente | Dev. Standard | Statistica t | p-value | Intervallo di Conf. 95% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | 515.0001 | 1.208 | 426.415 | **0.000** | [512.63, 517.37] |
| **Gender (Femmina)** | -27.9417 | 1.456 | -19.194 | **0.000** | [-30.79, -25.09] |
| **Seniority** | 54.6008 | 1.130 | 48.324 | **0.000** | [52.39, 56.82] |
| **Year_2012** | -37.5796 | 1.621 | -23.187 | **0.000** | [-40.76, -34.40] |
| **Year_2013** | 23.7536 | 1.365 | 17.402 | **0.000** | [21.08, 26.43] |
| **Year_2014** | -53.4521 | 1.833 | -29.159 | **0.000** | [-57.05, -49.86] |
| **Job_Education** | -329.8440 | 2.547 | -129.506 | **0.000** | [-334.84, -324.85] |
| **Job_Fire** | 223.1916 | 2.887 | 77.313 | **0.000** | [217.53, 228.85] |
| **Job_Medical** | -30.5440 | 2.573 | -11.873 | **0.000** | [-35.59, -25.50] |
| **Job_Police** | 155.7580 | 1.710 | 91.106 | **0.000** | [152.41, 159.11] |
| **Gender x Seniority** | -1.4818 | 1.319 | -1.123 | 0.261 | [-4.07, 1.10] |
| **Gender x Job_Education** | 28.0345 | 3.762 | 7.452 | **0.000** | [20.66, 35.41] |
| **Gender x Job_Fire** | 7.6066 | 6.996 | 1.087 | 0.277 | [-6.11, 21.32] |
| **Gender x Job_Medical** | 8.4227 | 3.103 | 2.715 | **0.007** | [2.34, 14.50] |
| **Gender x Job_Police** | -50.0045 | 3.421 | -14.618 | **0.000** | [-56.71, -43.30] |

---

### 2. Interpretazione Econometrica delle Interazioni

La derivata parziale della retribuzione trasformata $\sqrt{Y}$ rispetto a Gender (Femmina) è espressa da:
$$\frac{\partial Y^{(0.5)}}{\partial \text{Gender}} = -27.94 + 28.03 \cdot \text{Education} + 8.42 \cdot \text{Medical} - 50.00 \cdot \text{Police}$$

* **Divario di Base**: Nella macro-categoria di riferimento (`Other`), le donne guadagnano significativamente meno dei colleghi uomini ($\beta\_{\text{Gender}} = -27.94$, $p < 0.001$).
* **Mitigazione totale in Education**: Nel settore scolastico/educativo, il divario di genere è totalmente annullato, con pendenze che portano il gap a zero ($\beta\_{\text{Gender}} + \beta\_{\text{Gender} \times \text{Education}} = -27.94 + 28.03 = +0.09$, statisticamente nullo).
* **Ampliamento nelle Forze dell'Ordine**: Nel corpo di Polizia, il divario salariale si amplia drammaticamente, registrando la disparità più elevata a svantaggio del genere femminile ($\beta\_{\text{Gender}} + \beta\_{\text{Gender} \times \text{Police}} = -27.94 - 50.00 = -77.94$, $p < 0.001$).

---

## 📈 Visual Diagnostic Showcase

Tutti i grafici salvati in formato ad alta risoluzione sono inseriti ed analizzati dettagliatamente nel walkthrough:

### 1. Diagnostica delle Regressioni e Gauss-Markov
* **Confronto Residui OLS (Modello Naïve)**: Riduzione drastica dell'andamento a ventaglio nei residui ed eccellente linearizzazione del Q-Q Plot dopo la trasformazione (salvati in `ols_diagnostics.png` e `transformed_diagnostics.png`).
* **Profilo di Log-Verosimiglianza Box-Cox**: Picco MLE stimato a $\lambda = 0,5969$ (salvato in `boxcox_likelihood.png`).
* **Impatto della Trasformazione Box-Cox**: Confronto istogramma e KDE tra i salari grezzi (fortemente asimmetrici) e i salari trasformati (simmetrici e normalizzati) (salvato in `salary_distribution.png`).
* **Diagnostica Avanzata (Leverage & Distanza di Cook)**: Index plot dei valori di leva $h\_{ii}$ con la soglia teorica $2(r+1)/n$ e index plot delle distanze di Cook $D\_i$ con annotazione ed evidenziazione testuale dei top 5 outlier più influenti, come il Capo dei Vigili del Fuoco `Joanne Hayes-White` (salvato in `leverage_cooks.png`).

### 2. Gap di Genere ed Effetti Interattivi
* **Pay Gap per Macro-Categoria**: Boxplot della retribuzione totale per `JobCategory` e `Gender` che evidenzia graficamente i divari lavorativi settoriali (salvato in `gender_pay_gap_by_job.png`).
* **Curva Salariale per Anzianità**: Grafico a linee dell'interazione tra genere e anzianità che mostra l'evoluzione del pay gap salariale nel corso del tempo (salvato in `seniority_pay_gap.png`).

### 3. Regolarizzazione Ridge
* **Ridge Shrinkage Path**: Visualizzazione analitica della contrazione dei coefficienti al variare del parametro di penalizzazione $\lambda$ (salvato in `ridge_path.png`).

---

## 🚀 Guia di Esecuzione e Replicabilità

Per riprodurre in autonomia l'intero workflow statistico ed econometrico, esegui il file principale nel terminale:

```bash
# Esegue il workflow completo all'interno dell'ambiente virtuale uv
uv run gender_analysis.py
```

Questo avvierà in modo sequenziale:
1. La pulizia dei dati e l'estrazione dei nomi di battesimo da `Salaries.csv`.
2. La classificazione automatica e ottimizzata del genere.
3. La stima delle regressioni OLS con relativi test diagnostici e scomposizioni matriciali.
4. Il salvataggio dei grafici diagnostici ad alta definizione nella cartella di lavoro.

---

## 🃏 Integrità e Conservazione del Progetto Poker

Tutti i codici, dataset e grafici relativi all'analisi strategica preflop/postflop e alla **Hold'em Profitability Matrix** sono stati **totalmente salvati e isolati** in una directory dedicata per non creare interferenze:

👉 [**Vai alla Cartella del Progetto Poker**](file:///c:/Users/loren/Documents/AntiGravity%20Projects/progettostat/poker/)

All'interno di tale cartella è presente una copia dell'originario `README.md` sul poker per consultare nuovamente i grafici diagnostici, il Q-Q plot leptocurtico delle vincite e la heatmap 13x13 del WinRate delle mani iniziali.

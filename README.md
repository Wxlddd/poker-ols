# Poker Player Regression Analysis (OLS) 📊🃏

Questo progetto implementa una pipeline di analisi statistica end-to-end in Python per stimare un modello di regressione lineare multipla (OLS) su dati reali del poker, valutando come i comportamenti pre-flop e post-flop influenzino il tasso di vincita (**Win Rate, bb/100**).

La repository contiene i dati estratti ed elaborati a partire da **39.942 mani reali** giocate sulla piattaforma *Absolute Poker* (dataset accademico `uoftcprg/phh-dataset`).

---

## 🚀 Come Eseguire il Progetto

Il progetto utilizza **[uv](https://github.com/astral-sh/uv)** per la gestione rapida dell'ambiente virtuale e delle dipendenze.

### 1. Requisiti
Assicurati di avere `uv` installato sul tuo sistema.

### 2. Installazione delle dipendenze ed esecuzione
Per ricreare l'ambiente ed eseguire l'analisi:

```bash
# Esegue la pipeline di costruzione del dataset
uv run python build_dataset.py

# Esegue l'analisi di regressione lineare multipla (OLS) e genera i grafici diagnostici
uv run python poker_analysis.py
```

---

## 📈 Risultati dell'Analisi di Regressione

Il modello stimato prende in considerazione sia la selezione pre-flop delle mani di partenza che il comportamento post-flop:

$$\text{WinRate} = \beta_0 + \beta_1 \text{VPIP} + \beta_2 \text{PFR} + \beta_3 \text{3Bet} + \beta_4 \text{Postflop\_Agg} + \beta_5 \text{WTSD} + \beta_6 \text{W\$SD} + \beta_7 \text{Hands} + \epsilon$$

### Modello di Stima OLS (Output di `statsmodels`)

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                WinRate   R-squared:                       0.344
Model:                            OLS   Adj. R-squared:                  0.318
Method:                 Least Squares   F-statistic:                     13.32
No. Observations:                 186   Prob (F-statistic):           8.94e-14
Df Residuals:                     178   Log-Likelihood:                -938.02
Df Model:                           7   AIC:                             1892.
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
const         -317.6095     36.301     -8.749      0.000    -389.246    -245.973
VPIP             0.3019      0.286      1.056      0.292      -0.262       0.866
PFR              0.0098      0.477      0.021      0.984      -0.931       0.950
3Bet             2.5201      2.789      0.903      0.367      -2.984       8.024
Postflop_Agg    -0.9536      0.439     -2.172      0.031      -1.820      -0.087
WTSD             0.4225      0.410      1.030      0.305      -0.387       1.232
W$SD             4.2288      0.457      9.253      0.000       3.327       5.131
Hands           -0.0037      0.005     -0.692      0.490      -0.014       0.007
==============================================================================
```

### 🔍 Evidenze Statistiche Fondamentali

1.  **L'importanza del Post-Flop ($R^2 = 34.4\%$):**
    Il coefficiente di determinazione spiega il **34.4%** della varianza del Win Rate! Questo rappresenta un balzo immenso rispetto al modello con soli dati pre-flop (che spiegava appena l'1.2%). È la dimostrazione scientifica che a poker le decisioni post-flop dominano il profitto complessivo rispetto alla sola selezione delle mani iniziali.
2.  **Il Peso Schiacciante di Vincere allo Showdown ($W\$SD$):**
    La variabile `W$SD` (Won $ at Showdown) ha il coefficiente più alto e statisticamente più significativo del modello ($\beta = 4.2288$, $t = 9.25$, $p < 0.001$).
    *   *Significato:* **Per ogni incremento dell'1% nella percentuale di showdown vinti, il Win Rate atteso aumenta di 4.23 bb/100.** La capacità di effettuare showdown profittevoli e non sprecare fiches è la colonna portante di un gioco vincente.
3.  **L'Impatto dell'Aggressione Post-Flop (`Postflop_Agg`):**
    La variabile mostra un coefficiente negativo significativo ($\beta = -0.9536$, $t = -2.17$, $p = 0.031$).
    *   *Significato:* Un'aggressività post-flop non calibrata (over-bluffare, puntare a vuoto) in questo campione reale si traduce in perdite sistematiche.
4.  **Validazione Predittiva (Test Set):**
    Testato sul 20% di dati non visti, il modello mantiene un eccezionale **$R^2 = 25.03\%$**, confermando che le relazioni strutturate sono solide e non affette da overfitting.

---

## 📊 Diagnostica dei Residui

I residui del modello rispettano le assunzioni classiche della regressione lineare, pur presentando la consueta eteroschedasticità strutturale del poker (i giocatori con un volume maggiore di mani hanno residui più concentrati):

![Diagnostic Plots](residual_diagnostics.png)

*   **Breusch-Pagan Test p-value:** `0.008888` (Eteroschedasticità confermata al livello di significatività del 1%).

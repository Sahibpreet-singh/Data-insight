# Crypto Trading Sentiment Analysis

> Exploring the relationship between the Fear & Greed Index and trader profitability across historical crypto trades.

---

## Overview

This project investigates how prevailing market sentiment — as measured by the **Crypto Fear & Greed Index** — correlates with trading behavior and profitability. Using a historical dataset of individual trades merged with daily sentiment classifications, the analysis identifies patterns across market conditions, profiles trader archetypes, and applies unsupervised clustering to segment the trader population.

---

## Project Structure

```
.
├── datasets/
│   ├── fear_greed_index.csv      # Daily Fear & Greed Index with sentiment classification
│   └── historical_data.csv       # Historical trade records (account, coin, PnL, size, etc.)
├── main.ipynb                    # Main analysis notebook
└── README.md
```

---

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **Data Loading** | Load trade and sentiment datasets; inspect raw timestamp formats |
| 2 | **Data Cleaning** | Normalise timestamps, extract date keys, check for missing values |
| 3 | **Data Merging** | Inner join trades with sentiment on date; validate merged output |
| 4 | **Exploratory Analysis** | PnL distribution, feature correlation heatmap, trade counts by sentiment |
| 5 | **Sentiment vs Profitability** | Mean/total PnL and position size breakdowns by sentiment classification |
| 6 | **Trader Behavior Analysis** | Per-coin PnL, best/worst account deep-dive, sentiment distribution comparison |
| 7 | **Clustering Analysis** | K-Means segmentation of traders by profitability, size, and activity |
| 8 | **Key Insights** | Summary table of principal findings |
| 9 | **Conclusion** | Synthesis, limitations, and recommended next steps |

---

## Datasets

### `fear_greed_index.csv`
Daily sentiment readings for the crypto market.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | Unix (seconds) | Date of the sentiment reading |
| `value` | Integer | Raw Fear & Greed score (0–100) |
| `classification` | String | Label: *Extreme Fear, Fear, Neutral, Greed, Extreme Greed* |

### `historical_data.csv`
Individual trade records with account, asset, and outcome information.

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | Unix (milliseconds) | Trade execution time |
| `Account` | String | Trader account identifier |
| `Coin` | String | Traded asset (e.g. BTC, ETH) |
| `Execution Price` | Float | Price at which the trade was executed |
| `Size Tokens` | Float | Trade size in token units |
| `Size USD` | Float | Trade size in USD |
| `Closed PnL` | Float | Realised profit or loss on the trade |
| `Fee` | Float | Transaction fee paid |

---

## Key Findings

- **Sentiment and trade volume** — Trading activity was not uniformly distributed across sentiment conditions, suggesting sentiment-driven participation patterns.
- **Sentiment and profitability** — Higher average profitability was observed during specific sentiment classifications; however, correlation does not imply a causal mechanism.
- **Position sizing behaviour** — Average position size varied across sentiment labels, indicating that traders adjust risk exposure in response to market conditions.
- **Account performance dispersion** — A substantial gap exists between the highest and lowest total PnL accounts, reflecting significant heterogeneity in trader skill or strategy.
- **Clustering reveals distinct archetypes** — K-Means clustering (k=3) identified three trader profiles differentiated by profitability, position size, and activity level.
- **Right-skewed PnL distribution** — The majority of trades cluster near zero; overall profitability is driven by a small number of high-gain trades.

---

## Requirements

```
pandas
matplotlib
seaborn
scikit-learn
```

Install all dependencies with:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## Usage

1. Clone the repository and place the datasets in the `datasets/` folder.
2. Launch Jupyter:
   ```bash
   jupyter notebook main.ipynb
   ```
3. Run all cells in order (`Kernel → Restart & Run All`).

---

## Limitations

- The inner join excludes trading days without a matching sentiment record, which may introduce temporal gaps.
- Cluster count (k=3) was selected without a formal elbow-method or silhouette-score evaluation.
- The Fear & Greed Index is a lagging, coarse-grained signal; intraday or alternative sentiment proxies could improve resolution.
- Statistical significance tests (e.g., ANOVA, Kruskal–Wallis) should be applied to confirm that observed PnL differences across sentiment conditions are not attributable to sampling variation.

---

## License

This project is for educational and analytical purposes.

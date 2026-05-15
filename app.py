import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Crypto Sentiment Dashboard", layout="wide")
st.title("📊 Crypto Trading Sentiment Dashboard")
st.caption("Analyzing how the Fear & Greed Index affects trading profitability.")


# ── Load data ─────────────────────────────────────────────────────────────────

sentiment = pd.read_csv("datasets/fear_greed_index.csv")
trades    = pd.read_csv("datasets/historical_data.csv")


# ── Convert timestamps ────────────────────────────────────────────────────────

sentiment["timestamp"] = pd.to_datetime(sentiment["timestamp"], unit="s")
trades["Timestamp"]    = pd.to_datetime(trades["Timestamp"], unit="ms")

sentiment["date"] = sentiment["timestamp"].dt.date
trades["date"]    = trades["Timestamp"].dt.date


# ── Merge datasets ────────────────────────────────────────────────────────────

df = pd.merge(trades, sentiment[["date", "classification"]], on="date", how="inner")

df["is_profitable"] = (df["Closed PnL"] >= 0).astype(int)


# ── Sidebar filters ───────────────────────────────────────────────────────────

st.sidebar.header("Filters")

all_sentiments  = sorted(df["classification"].dropna().unique().tolist())
selected_sent   = st.sidebar.multiselect("Sentiment", all_sentiments, default=all_sentiments)

all_coins       = sorted(df["Coin"].dropna().unique().tolist())
selected_coins  = st.sidebar.multiselect("Coin (blank = all)", all_coins, default=[])

# Apply filters
if selected_coins:
    df = df[df["classification"].isin(selected_sent) & df["Coin"].isin(selected_coins)]
else:
    df = df[df["classification"].isin(selected_sent)]

if df.empty:
    st.warning("No data for selected filters.")
    st.stop()


# ── KPIs ──────────────────────────────────────────────────────────────────────

st.subheader("Key Metrics")

total_trades = len(df)
avg_pnl      = df["Closed PnL"].mean()
total_pnl    = df["Closed PnL"].sum()
win_rate     = df["is_profitable"].mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades",  f"{total_trades:,}")
col2.metric("Average PnL",   f"${avg_pnl:,.2f}")
col3.metric("Total PnL",     f"${total_pnl:,.0f}")
col4.metric("Win Rate",      f"{win_rate:.1f}%")

st.divider()


# ── Chart 1 — Trade count by sentiment ───────────────────────────────────────

st.subheader("Sentiment Analysis")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Trade Count by Sentiment**")

    trade_count = df.groupby("classification").size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(trade_count.index, trade_count.values, color="#4C72B0")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Trades")
    ax.set_title("Trades per Sentiment")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Chart 2 — Average PnL by sentiment ───────────────────────────────────────

with col_b:
    st.markdown("**Average PnL by Sentiment**")

    avg_pnl_sent = df.groupby("classification")["Closed PnL"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(avg_pnl_sent.index, avg_pnl_sent.values, color="#55A868")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Avg Closed PnL (USD)")
    ax.set_title("Avg Profit per Sentiment")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()


# ── Chart 3 — PnL histogram ───────────────────────────────────────────────────

st.subheader("PnL Distribution")
st.caption("Most trades are near zero. Long tails are outliers.")

clipped = df["Closed PnL"].clip(-2000, 2000)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.hist(clipped, bins=60, color="#4C72B0", edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Break-even")
ax.set_xlabel("Closed PnL (USD)  [clipped at ±$2,000]")
ax.set_ylabel("Number of Trades")
ax.set_title("PnL Distribution")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()


# ── Chart 4 — Correlation heatmap ─────────────────────────────────────────────

st.subheader("Feature Correlation")
st.caption("How Size USD, PnL, and Fee relate to each other.")

corr_data   = df[["Size USD", "Closed PnL", "Fee"]].corr()

fig, ax = plt.subplots(figsize=(5, 3.5))
sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()


# ── Top 5 traders ─────────────────────────────────────────────────────────────

st.subheader("Top 5 Traders by Total PnL")

top5 = df.groupby("Account")["Closed PnL"].sum().sort_values(ascending=False).head(5).reset_index()
top5.columns = ["Account", "Total PnL (USD)"]
top5["Total PnL (USD)"] = top5["Total PnL (USD)"].round(2)
top5.index = top5.index + 1

st.dataframe(top5, use_container_width=True)

st.divider()
st.caption("Data: Hyperliquid trades × Alternative.me Fear & Greed Index")
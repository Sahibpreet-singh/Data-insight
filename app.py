import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Crypto Sentiment Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F1117;
    color: #C8CCD8;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #E0E3EF;
}
div[data-testid="stSidebar"] {
    background-color: #0D0F18;
    border-right: 1px solid #1E2130;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Crypto Sentiment Dashboard")
st.caption("Analyzing how the Fear & Greed Index affects trading profitability.")


# ── Matplotlib dark style ─────────────────────────────────────────────────────

plt.rcParams["figure.facecolor"]  = "#0F1117"
plt.rcParams["axes.facecolor"]    = "#1A1D27"
plt.rcParams["axes.edgecolor"]    = "#2E3140"
plt.rcParams["axes.labelcolor"]   = "#C8CCD8"
plt.rcParams["xtick.color"]       = "#8C8C8C"
plt.rcParams["ytick.color"]       = "#8C8C8C"
plt.rcParams["text.color"]        = "#E0E3EF"
plt.rcParams["axes.titlecolor"]   = "#E0E3EF"
plt.rcParams["axes.titleweight"]  = "bold"


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

st.sidebar.header("⚙️ Filters")

all_sentiments = sorted(df["classification"].dropna().unique().tolist())
selected_sent  = st.sidebar.multiselect("Sentiment", all_sentiments, default=all_sentiments)

all_coins      = sorted(df["Coin"].dropna().unique().tolist())
selected_coins = st.sidebar.multiselect("Coin (blank = all)", all_coins, default=[])

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


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Sentiment Overview",
    "💰  Profitability",
    "👤  Trader Profiles",
    "🔬  Statistics",
])


# ── Tab 1 — Sentiment Overview ────────────────────────────────────────────────

with tab1:
    st.subheader("How trading activity distributes across market sentiment")

    col_a, col_b = st.columns(2)

    # Chart 1 — Trade count by sentiment
    with col_a:
        st.markdown("**Trade Count by Sentiment**")

        trade_count = df.groupby("classification").size().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(trade_count.index, trade_count.values, color="#4C72B0", edgecolor="none")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Trades")
        ax.set_title("Trades per Sentiment Label")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2 — Win rate by sentiment
    with col_b:
        st.markdown("**Win Rate by Sentiment**")

        win_rate_sent = df.groupby("classification")["is_profitable"].mean() * 100

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(win_rate_sent.index, win_rate_sent.values, color="#DD8452", edgecolor="none")
        ax.axhline(50, color="white", linestyle="--", linewidth=0.8, label="50% baseline")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate by Sentiment")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Summary table
    st.markdown("#### Sentiment Summary Table")

    summary = df.groupby("classification").agg(
        Trades     = ("Account",       "count"),
        Mean_PnL   = ("Closed PnL",    "mean"),
        Total_PnL  = ("Closed PnL",    "sum"),
        Avg_Size   = ("Size USD",      "mean"),
    ).round(2)

    st.dataframe(summary, use_container_width=True)


# ── Tab 2 — Profitability ─────────────────────────────────────────────────────

with tab2:
    st.subheader("PnL distributions and position sizing by market sentiment")

    col_a, col_b = st.columns(2)

    # Chart 3 — PnL histogram
    with col_a:
        st.markdown("**PnL Distribution**")

        clipped = df["Closed PnL"].clip(-2000, 2000)

        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.hist(clipped, bins=60, color="#4C72B0", edgecolor="none", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Break-even")
        ax.set_xlabel("Closed PnL (USD)  [clipped at ±$2,000]")
        ax.set_ylabel("Number of Trades")
        ax.set_title("PnL Distribution")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 4 — Avg position size by sentiment
    with col_b:
        st.markdown("**Average Position Size by Sentiment**")

        avg_size = df.groupby("classification")["Size USD"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.bar(avg_size.index, avg_size.values, color="#55A868", edgecolor="none")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Avg Size (USD)")
        ax.set_title("Average Position Size")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 5 — Avg PnL by sentiment
    st.markdown("**Average PnL by Sentiment**")

    avg_pnl_sent = df.groupby("classification")["Closed PnL"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(avg_pnl_sent.index, avg_pnl_sent.values, color="#C44E52", edgecolor="none")
    ax.axhline(0, color="white", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Avg Closed PnL (USD)")
    ax.set_title("Average Profit per Trade by Sentiment")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Tab 3 — Trader Profiles ───────────────────────────────────────────────────

with tab3:
    st.subheader("Top and bottom traders")

    # Top 5 traders
    st.markdown("#### Top 5 Traders by Total PnL")

    top5 = df.groupby("Account")["Closed PnL"].sum().sort_values(ascending=False).head(5).reset_index()
    top5.columns = ["Account", "Total PnL (USD)"]
    top5["Total PnL (USD)"] = top5["Total PnL (USD)"].round(2)
    top5.index = top5.index + 1

    st.dataframe(top5, use_container_width=True)

    st.divider()

    # Best vs worst account sentiment breakdown
    st.markdown("#### Best vs Worst Account — Sentiment Mix")

    account_pnl   = df.groupby("Account")["Closed PnL"].sum()
    best_account  = account_pnl.idxmax()
    worst_account = account_pnl.idxmin()

    best_trades  = df[df["Account"] == best_account]
    worst_trades = df[df["Account"] == worst_account]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Best Account** — Total PnL: ${account_pnl[best_account]:,.0f}")

        best_sent = best_trades["classification"].value_counts()

        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        ax.bar(best_sent.index, best_sent.values, color="#4C72B0", edgecolor="none")
        ax.set_title("Best Account — Trades by Sentiment")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=25, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown(f"**Worst Account** — Total PnL: ${account_pnl[worst_account]:,.0f}")

        worst_sent = worst_trades["classification"].value_counts()

        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        ax.bar(worst_sent.index, worst_sent.values, color="#C44E52", edgecolor="none")
        ax.set_title("Worst Account — Trades by Sentiment")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=25, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── Tab 4 — Statistics ────────────────────────────────────────────────────────

with tab4:
    st.subheader("Descriptive statistics and correlation")

    # Descriptive stats
    st.markdown("#### PnL Stats by Sentiment")

    desc = df.groupby("classification")["Closed PnL"].describe().round(2)
    st.dataframe(desc, use_container_width=True)

    st.divider()

    # Correlation heatmap
    st.markdown("#### Feature Correlation Heatmap")
    st.caption("Shows how Size USD, PnL, and Fee relate to each other.")

    corr_data = df[["Size USD", "Closed PnL", "Fee"]].corr()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("Data: Hyperliquid trades × Alternative.me Fear & Greed Index")
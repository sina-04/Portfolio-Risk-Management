# Portfolio Management on Tehran Stock Exchange (TSE)

## Introduction

This repository contains a Jupyter Notebook for analyzing and optimizing stock portfolios on the Tehran Stock Exchange (TSE).

It guides you from **data collection** (stocks, sector indices, USD/Rial, and multiple symbols) to **risk/return analysis**, **advanced risk measures (CDaR)**, and **portfolio optimization** using **Riskfolio**, with **Persian-friendly plots** (right-to-left labels and Farsi fonts).

The notebook works with **Jalali dates (1394–1404)** and demonstrates how to prepare TSE data for quantitative portfolio analysis.

---

## Features

- 📥 **Data download from TSE** via `finpy_tse` and `tse_option`

  - Individual stocks (e.g. `رتاپ`, `البرز`)
  - Sector indices (e.g. خودرو, فلزات اساسی)
  - Total market indices: Capital-Weighted Index (CWI) and Equal-Weighted Index (EWI)
  - Multiple symbols at once via `tse_option.download`

- 📊 **Return and risk analysis**

  - Daily simple returns
  - Standard deviation (volatility)
  - Mean Absolute Deviation (MAD)
  - Conditional Drawdown at Risk (CDaR) for individual stocks and portfolios
  - Correlations between stocks, sectors, and USD

- 📈 **Portfolio optimization using Riskfolio**

  - Historical (Classic) mean-variance framework
  - Flexible risk measures (`MV`, `MAD`, `CVaR`, `CDaR`, etc.)
  - `MinRisk`, `MaxRet`, `Sharpe`, and `Utility` objectives (example uses `MinRisk`)
  - Calculation of portfolio expected return, volatility, and CDaR

- 🎨 **Persian-friendly visualization**

  - Right-to-left reshaping for Persian text using `python-bidi` and `arabic-reshaper`
  - Custom font (`Vazir-Bold.ttf`)
  - Pie chart of optimal portfolio weights with Persian labels
  - Yearly normalized line chart (Jalali years) of USD vs sector indices

- 🧹 **Data preparation & cleaning**

  - Handling `J-Date` columns as time index
  - De-duplicating dates and sorting
  - Reindexing to business days and forward-filling missing values
  - Aligning multiple time series to a common date range

---

## System Requirements

- **Python**: 3.8+ (recommended)
- **Jupyter Notebook** or JupyterLab
- **Libraries** (install via `pip`):

  - `pandas`
  - `numpy`
  - `matplotlib`
  - `finpy_tse`
  - `tse_option`
  - `riskfolio-lib` (imported as `riskfolio`)
  - `arabic-reshaper`
  - `python-bidi`

- **Fonts**:

  - `Vazir-Bold.ttf` placed in the same folder as the notebook (or another Persian font you configure)

- **Internet access**:

  - Required for downloading up-to-date TSE data via `finpy_tse` and `tse_option`

---

## Installation

Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
pip install pandas numpy matplotlib finpy_tse tse_option riskfolio-lib arabic-reshaper python-bidi
```

Place your Persian font file (e.g. `Vazir-Bold.ttf`) in the same directory as the notebook, or adjust the path in the code.

Clone this repository and open the notebook:

```bash
git clone https://github.com/sina-04/Portfolio-Risk-Management.git
cd Portfolio-Risk-Management
jupyter notebook "Portfolio Management Jupyter Notebook.ipynb"
```

---

## Usage

### 1. Imports and Matplotlib Configuration

The notebook starts by importing required libraries and configuring Matplotlib for high-resolution plots:

```python
%matplotlib inline
%config InlineBackend.figure_format = "retina"

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import finpy_tse as fpy
import tse_option as tso
import riskfolio as rp

mpl.rcParams["figure.dpi"] = 150       # resolution in notebook/window
mpl.rcParams["savefig.dpi"] = 300      # resolution when saving
```

### 2. Global Configuration and Symbols

Jalali date range and symbol mappings are defined in a config cell:

```python
J_START = "1394-09-18"
J_END   = "1404-09-18"

symbols = {
    "retap": "رتاپ",
    "alborz": "البرز",
}

sectors = {
    "auto": "خودرو",
    "base_metals": "فلزات اساسی",
}
```

You can customize these to use different stocks or sectors.

### 3. Helper Functions

Two helper functions centralize common tasks:

```python
def get_price_history(stock, start=J_START, end=J_END, adj=False):
    df = fpy.Get_Price_History(
        stock=stock,
        start_date=start,
        end_date=end,
        ignore_date=False,
        adjust_price=adj,
        show_weekday=False,
        double_date=False,
    )
    return df

def get_tse_option(symbols, start, end=None, j_date=True):
    return tso.download(
        symbols=symbols,
        j_date=j_date,
        start=start,
        end=end,
        adjust_price=True,
        drop_unadjusted=False,
    )
```

These are used throughout for concise, repeatable data downloads.

---

## Examples

### Example 1 – Downloading Price History for Single Stocks

Download prices for two TSE stocks (`رتاپ` and `البرز`):

```python
# Stock of رتاپ
df1 = fpy.Get_Price_History(
    stock="رتاپ",
    start_date=J_START,
    end_date=J_END,
    ignore_date=False,
    adjust_price=False,   # or True for adjusted prices
    show_weekday=False,
    double_date=False
)

# Stock of البرز
df2 = fpy.Get_Price_History(
    stock="البرز",
    start_date=J_START,
    end_date=J_END,
    ignore_date=False,
    adjust_price=False,
    show_weekday=False,
    double_date=False
)
```

Typical output: a `DataFrame` indexed by Jalali trading date, with columns such as `Open`, `High`, `Low`, `Close`, volume, etc.

---

### Example 2 – Total Market Indices (CWI & EWI)

Retrieve capital-weighted and equal-weighted total market indices:

```python
df_cwi = fpy.Get_CWI_History(
    start_date='1395-01-01',
    end_date='1400-12-29',
    ignore_date=False,
    just_adj_close=False,
    show_weekday=False,
    double_date=False
)

df_ewi = fpy.Get_EWI_History(
    start_date='1395-01-01',
    end_date='1400-12-29',
    ignore_date=False,
    just_adj_close=True,
    show_weekday=False,
    double_date=False
)

display(df_cwi)
display(df_ewi)
```

This gives you benchmark series for the whole market.

---

### Example 3 – Downloading Multiple Symbols with `tse_option`

Use `tse_option` to download historical data for one or several symbols:

```python
# Single symbol example
df52 = tso.download(
    symbols=["ارفع"],
    j_date=True,
    start="1400-05-20",
    end="1402-11-23",
    adjust_price=True,
    drop_unadjusted=False,
)
display(df52)

# Multiple symbols example
df53 = tso.download(
    symbols=["شلرد", "زماهان", "نطرین", "شپنا", "برکت"],
    j_date=True,
    start="1399-05-20",
    end=None,
    adjust_price=True,
    drop_unadjusted=False,
)
display(df53)
```

The returned `DataFrame` includes pricing information for each symbol, suitable for further filtering and analysis.

---

### Example 4 – Building a Two-Stock Return Series and Correlation

Merge the two stocks into a single `DataFrame` and compute daily returns and correlation:

```python
# Rename price columns to stock names
df1_prices = df1.rename(columns={"Close": "رتاپ"})
df2_prices = df2.rename(columns={"Close": "البرز"})

# Align on common index
merged_df0 = pd.concat(
    [df1_prices["رتاپ"], df2_prices["البرز"]],
    axis=1
)

# Daily simple returns
returns = merged_df0.pct_change().dropna()

# Correlation between the two stocks
corr_value = returns["رتاپ"].corr(returns["البرز"])
print(f"Correlation between رتاپ and البرز: {corr_value:.5f}")
```

**Output**: a single scalar correlation coefficient printed in the console.

---

### Example 5 – Basic Risk Metrics: Std, MAD, and CDaR (Single Stock)

Compute volatility, MAD, and CDaR for each stock:

```python
# Standard deviation (volatility)
STD_retap  = np.std(returns["رتاپ"])
STD_alborz = np.std(returns["البرز"])

print(f"Std Dev رتاپ : {STD_retap:.5f}")
print(f"Std Dev البرز : {STD_alborz:.5f}")

# MAD (Mean Absolute Deviation)
retap_ret   = returns["رتاپ"]
alborz_ret  = returns["البرز"]
MAD_retap   = np.mean(np.abs(retap_ret  - retap_ret.mean()))
MAD_alborz  = np.mean(np.abs(alborz_ret - alborz_ret.mean()))

print(f"MAD رتاپ  : {MAD_retap:.5f}")
print(f"MAD البرز : {MAD_alborz:.5f}")
```

CDaR (Conditional Drawdown at Risk) per stock:

```python
# Returns for each stock
retap_ret  = returns["رتاپ"]
alborz_ret = returns["البرز"]

alpha = 0.95  # confidence level for CDaR

def calc_cdar(r, alpha=0.95):
    cum_wealth = (1 + r).cumprod()
    running_max = cum_wealth.cummax()
    drawdowns = 1 - (cum_wealth / running_max)
    threshold_dd = drawdowns.quantile(alpha)
    cdar = float(drawdowns[drawdowns >= threshold_dd].mean())
    return cdar

CDaR_retap  = calc_cdar(retap_ret,  alpha=alpha)
CDaR_alborz = calc_cdar(alborz_ret, alpha=alpha)

print(f"CDaR رتاپ  (alpha={alpha}) : {CDaR_retap:.5f}")
print(f"CDaR البرز (alpha={alpha}) : {CDaR_alborz:.5f}")
```

You also get reusable helpers:

```python
def compute_basic_stats(returns: pd.Series) -> pd.Series:
    return pd.Series({
        "mean": returns.mean(),
        "std": returns.std(),
        "mad": np.mean(np.abs(returns - returns.mean())),
    })

def cdar(returns: pd.Series, alpha: float = 0.95) -> float:
    cum_wealth = (1 + returns).cumprod()
    running_max = cum_wealth.cummax()
    drawdowns = 1 - (cum_wealth / running_max)
    threshold = drawdowns.quantile(alpha)
    return drawdowns[drawdowns >= threshold].mean()

stats = pd.DataFrame({
    "رتاپ": compute_basic_stats(returns["رتاپ"]),
    "البرز": compute_basic_stats(returns["البرز"]),
}).T
```

---

### Example 6 – Portfolio Optimization with Riskfolio

#### 6.1 Create `Portfolio` object

```python
# Building the portfolio object
port = rp.Portfolio(returns=returns)
```

#### 6.2 Estimate inputs and optimize (MinRisk, MV)

```python
# Select method and estimate input parameters:
method_mu  = 'hist'   # expected returns
method_cov = 'hist'   # covariance

# Estimate mean and covariance based on historical data
port.assets_stats(method_mu=method_mu, method_cov=method_cov)

# Optimization parameters
model = 'Classic'  # Classic, BL, FM, BL_FM
rm    = 'MV'       # risk measure (13 available)
obj   = 'MinRisk'  # MinRisk, MaxRet, Utility, Sharpe
hist  = True       # historical scenarios
rf    = 0          # risk-free rate
l     = 0          # risk aversion (for Utility)

w = port.optimization(
    model=model,
    rm=rm,
    obj=obj,
    rf=rf,
    l=l,
    hist=hist,
)

display(w)
```

#### 6.3 Available risk measures

The notebook lists the supported risk measures:

```text
Risk Measures available:

- 'MV':   Standard Deviation
- 'MAD':  Mean Absolute Deviation
- 'SLPM': Second Lower Partial Moment (Sortino Ratio)
- 'MSV':  Semi Standard Deviation
- 'FLPM': First Lower Partial Moment (Omega Ratio)
- 'CVaR': Conditional Value at Risk
- 'EVaR': Entropic Value at Risk
- 'WR':   Worst Realization (Minimax)
- 'MDD':  Maximum Drawdown (Calmar Ratio)
- 'ADD':  Average Drawdown
- 'CDaR': Conditional Drawdown at Risk
- 'EDaR': Entropic Drawdown at Risk
- 'UCI':  Ulcer Index
```

You can try different `rm` values (e.g. `'CVaR'`, `'CDaR'`) to see how the efficient portfolio changes.

#### 6.4 Portfolio metrics (return, volatility, CDaR)

Expected return and volatility using mean and covariance:

```python
# Mean and covariance from historical returns
mu    = returns.mean()
Sigma = returns.cov()

w_vec = w.values.reshape(-1, 1)

port_ret = float(mu @ w.values)
port_vol = float(np.sqrt(w_vec.T @ Sigma.values @ w_vec))

print(f"Portfolio expected daily return : {port_ret:.5f}")
print(f"Portfolio daily volatility (std): {port_vol:.5f}")
print("\nWeights:")
display(w)
```

Portfolio CDaR based on the portfolio return time series:

```python
weights = w.squeeze()  # 1-column DataFrame -> Series
port_ret_series = returns.mul(weights, axis=1).sum(axis=1)

port_ret = float(port_ret_series.mean())
alpha    = 0.95

cum_wealth = (1 + port_ret_series).cumprod()
running_max = cum_wealth.cummax()
drawdowns = 1 - (cum_wealth / running_max)

threshold_dd = drawdowns.quantile(alpha)
cdar = float(drawdowns[drawdowns >= threshold_dd].mean())

print(f"Portfolio expected daily return     : {port_ret:.5f}")
print(f"Portfolio daily CDaR (alpha={alpha}): {cdar:.5f}")
```

---

### Example 7 – Persian Pie Chart of Optimal Portfolio

The notebook uses `python-bidi`, `arabic-reshaper`, and a Persian font to show a right-to-left pie chart of the optimal weights.

```python
from bidi.algorithm import get_display
import arabic_reshaper
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

def reshape_rtl(text: str) -> str:
    if not text:
        return text
    return get_display(arabic_reshaper.reshape(text))

vazir_font = FontProperties(fname="Vazir-Bold.ttf", size=14)
title_fa   = "سبد بهینه (رتاپ & البرز)"

ax = rp.plot_pie(
    w=w,
    others=0.05,
    cmap="tab20",
    title=title_fa,
    nrow=10,
)

# Fix text in labels, legend, and title to RTL and use Farsi font
for text in ax.texts:
    txt = text.get_text()
    if any("\u0600" <= ch <= "\u06FF" for ch in txt):
        text.set_text(reshape_rtl(txt))
    text.set_fontproperties(vazir_font)

leg = ax.get_legend()
if leg is not None:
    for t in leg.get_texts():
        t_str = t.get_text()
        if any("\u0600" <= ch <= "\u06FF" for ch in t_str):
            t.set_text(reshape_rtl(t_str))
        t.set_fontproperties(vazir_font)

ax.set_title(reshape_rtl(title_fa), fontproperties=vazir_font)

fig = ax.get_figure()
fig.savefig("optimum_portfolio_pie.png", dpi=300, bbox_inches="tight")
plt.show()
```

**Output**: `optimum_portfolio_pie.png` — a Persian-labeled pie chart of optimal weights.

---

### Example 8 – Correlation Between USD and Sector Indices

The notebook examines how USD relates to sector indices (e.g. Auto and Base Metals):

```python
start_date = "1394-09-18"
end_date   = "1404-09-18"

usd, auto, base_metals = (
    fpy.Get_USD_RIAL(start_date=start_date, end_date=end_date, ignore_date=False, show_weekday=False, double_date=False),
    fpy.Get_SectorIndex_History(sector="خودرو",      start_date=start_date, end_date=end_date, ignore_date=False, just_adj_close=True, show_weekday=False, double_date=False),
    fpy.Get_SectorIndex_History(sector="فلزات اساسی", start_date=start_date, end_date=end_date, ignore_date=False, just_adj_close=True, show_weekday=False, double_date=False),
)

def prepare_df(df):
    df = df.copy()
    if "J-Date" in df.columns:
        df = df.set_index("J-Date")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df

usd        = prepare_df(usd)
auto       = prepare_df(auto)
base_metals = prepare_df(base_metals)

# Combine and rename Close columns, reindex to business days, forward fill
df = pd.concat(
    [
        usd["Close"].rename("USD"),
        auto["Close"].rename("Auto"),
        base_metals["Close"].rename("BaseMetals"),
    ],
    axis=1
)

df = df.asfreq("B").ffill()   # business days, forward fill
returns = df.pct_change().dropna()

corr_auto = returns["USD"].corr(returns["Auto"])
corr_bm   = returns["USD"].corr(returns["BaseMetals"])

print(f"Correlation (USD vs Auto):        {corr_auto:.5f}")
print(f"Correlation (USD vs BaseMetals): {corr_bm:.5f}")
```

The notebook then visualizes these correlations:

```python
plt.figure(figsize=(6, 4))
plt.bar(["Auto", "BaseMetals"], [corr_auto, corr_bm])
plt.ylabel("Correlation with USD (daily returns)")
plt.title("1394-09-18 to 1404-09-18: USD vs Auto & Base Metals")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()
```

---

### Example 9 – Yearly Normalized Trends (Jalali Years)

Finally, the notebook constructs a yearly index (normalized to 100 in the first year) to compare the long-term behaviour of USD and sector indices:

```python
# Create Jalali year labels from index and group
years = pd.Series(df.index, index=df.index).astype(str).str.slice(0, 4).astype(int)

df_yearly = df.groupby(years).last()
df_yearly = df_yearly.loc[1394:1404]

# Normalize to 100 at first year
df_yearly_norm = df_yearly / df_yearly.iloc[0] * 100

series_name_map = {
    "USD":        "دلار آزاد",
    "Auto":       "شاخص گروه خودرو",
    "BaseMetals": "شاخص گروه فلزات اساسی",
}

plt.figure(figsize=(10, 5))
for col in df_yearly_norm.columns:
    label_fa = series_name_map.get(col, col)
    plt.plot(df_yearly_norm.index, df_yearly_norm[col], marker="o", label=label_fa)

plt.xlabel("سال (هجری شمسی)")
plt.ylabel("شاخص نرمال شده (اولین سال = 100)")
plt.title("مقایسه روند سالانه دلار و گروه‌های خودرویی و فلزات اساسی")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
```

This helps visually compare multi-year performance across FX and sectors.

---

## Summary

This notebook can serve as a template for:

- Pulling TSE data for multiple instruments and sectors
- Computing risk/return statistics and more advanced measures like CDaR
- Building and optimizing portfolios with Riskfolio
- Creating high-quality, Persian-friendly visualizations

To adapt it to your needs, change the symbol lists, date ranges, and Riskfolio parameters, and extend the helper functions and plots as required.

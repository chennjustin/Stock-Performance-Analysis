# ESG Ratings vs. Stock Performance Analysis

This project explores the relationship between **Environmental, Social, and Governance (ESG) ratings** and **stock performance** in the U.S. equity market. Using tools from  **CAPM** and **yfinance**, we analyze differences in expected returns, volatility, and Sharpe ratios between high-ESG and low-ESG stock groups.

---
## 🗂️ Project Structure
esg-stock-analysis/  
├── data/  
│   └── [Stock CSV files]  
├── requirements.txt  
├── Investment_Analysis.py  
└── README.md  


## 📊 Project Overview

The analysis is divided into the following steps:

1. **Data Collection**
   - Use `yfinance` to fetch historical daily prices for 10 selected U.S. stocks: 5 with high ESG ratings and 5 with low ESG ratings.

2. **Return & Risk Calculation**
   - Compute the daily and annualized return and standard deviation (volatility) for each stock.
   - Visualize them on a return vs. risk scatter plot.

3. **Portfolio Simulation**
   - Simulate 5000 random portfolio weight combinations (for both high-ESG and low-ESG sets).
   - Calculate the expected return, risk, and Sharpe Ratio for each portfolio.

4. **Efficient Frontier & Optimal Portfolio**
   - Plot the efficient frontier for both ESG groups.
   - Identify and highlight the portfolio with the highest Sharpe Ratio.

5. **CAPM-Based Metrics**
   - Compute alpha and beta for each stock relative to the S&P 500 index as the market benchmark.
   - Compare systematic risk between high and low ESG portfolios.

6. **Comparison and Interpretation**
   - Compare return distributions, volatility, Sharpe ratios, and CAPM metrics across ESG groups.

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

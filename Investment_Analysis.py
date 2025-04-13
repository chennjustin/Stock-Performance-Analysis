# 1
# Calculate monthly returns for past 60 months for each company
import yfinance as yf
import pandas as pd
from scipy.stats.mstats import winsorize

# Our selected companies with good and poor ESG ratings
good_esg_tickers = ["AXP", "BLK", "V", "IBM", "AAPL"]# , "AVGO", "NFLX", "TGT", "DIS", "REGN"]
poor_esg_tickers = ["WFC", "CADE", "DAL", "TSN", "KHC"]# , "K", "META", "MMM", "AHCO", "CLF"]

# Function to calculate monthly returns and winsorize them
def calculate_monthly_returns(ticker):
    stock = yf.Ticker(ticker)
    daily_data = stock.history(period="5y")

    monthly_data = daily_data["Close"].resample("1ME").last()

    monthly_returns = monthly_data.diff() / monthly_data.shift(1)
    monthly_returns = monthly_returns.dropna()

    winsorized_returns = winsorize(monthly_returns, limits=[0.05, 0.05])
    return winsorized_returns

good_esg_data = []
for ticker in good_esg_tickers:
    winsorized_returns = calculate_monthly_returns(ticker)
    good_esg_data.append(pd.DataFrame({"Company": ticker, "Monthly Return": winsorized_returns, }))

poor_esg_data = []
for ticker in poor_esg_tickers:
    winsorized_returns = calculate_monthly_returns(ticker)
    poor_esg_data.append(pd.DataFrame({"Company": ticker, "Monthly Return": winsorized_returns}))

# Prepare the DataFrames for better formatting
good_esg_df = pd.concat(good_esg_data, ignore_index=True)
poor_esg_df = pd.concat(poor_esg_data, ignore_index=True)

# Output
print("[Good ESG Rating Companies]")
print(good_esg_df)

print("\n[Poor ESG Rating Companies]")
print(poor_esg_df)


# 2
# Calculate expected return and standard deviation for each stock
import yfinance as yf
import pandas as pd
from scipy.stats.mstats import winsorize

# Function to calculate monthly returns and winsorize them
def calculate_metrics(ticker):
    stock = yf.Ticker(ticker)
    daily_data = stock.history(period="5y")

    monthly_data = daily_data["Close"].resample("1ME").last()

    monthly_returns = monthly_data.diff() / monthly_data.shift(1)
    monthly_returns = monthly_returns.dropna()

    winsorized_returns = winsorize(monthly_returns, limits=[0.05, 0.05])
    winsorized_returns = pd.Series(winsorized_returns, index=monthly_returns.index)

    expected_return = winsorized_returns.mean() * 12 # Annualized return
    std = winsorized_returns.std() * (12 ** 0.5) # Annualized std

    return winsorized_returns, round(expected_return, 4), round(std, 4)

# Create a list of dictionaries to store the calculated metrics(good ESG)
good_esg_data = []
for ticker in good_esg_tickers:
    winsorized_returns, expected_return, std = calculate_metrics(ticker)
    good_esg_data.append({
        "Company": ticker,
        "Expected Return": expected_return,
        "Standard Deviation": std
    })

# Create a list of dictionaries to store the calculated metrics(poor ESG)
poor_esg_data = []
for ticker in poor_esg_tickers:
    winsorized_returns, expected_return, std = calculate_metrics(ticker)
    poor_esg_data.append({
        "Company": ticker,
        "Expected Return": expected_return,
        "Standard Deviation": std
    })

# Create DataFrames for better formatting
good_esg_df = pd.DataFrame(good_esg_data)
poor_esg_df = pd.DataFrame(poor_esg_data)

# Output
print("[Good ESG Rating Companies]")
print(good_esg_df)

print("\n[Poor ESG Rating Companies]")
print(poor_esg_df)

# 3
# Constructing portfolios for good and poor ESG ratings
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to generate portfolios
def generate_portfolios(tickers, n_portfolios=100000, rf=0.02):
    returns_list = []
    for ticker in tickers:
        _, expected_return, _ = calculate_metrics(ticker)
        returns_list.append(expected_return)

    # Calculate covariance matrix
    return_data = [yf.Ticker(ticker).history(period="5y")["Close"].resample("1ME").last().pct_change().dropna() for ticker in tickers]
    return_df = pd.DataFrame(return_data).T
    return_df.columns = tickers
    covariance_matrix = return_df.cov()

    # Output covariance matrix
    print(f"\n[Covariance Matrix for {'Good ESG' if tickers == good_esg_tickers else 'Poor ESG'}]")  # For better traceability
    print(covariance_matrix.round(4))

    # Initialize lists to store portfolio metrics
    portfolio_returns = []
    portfolio_volatility = []
    portfolio_weights = []
    portfolio_sharpe = []

    # Generate random portfolios
    for _ in range(n_portfolios):
        # Randomly assign weights to each stock in the portfolio
        weights = np.random.dirichlet(np.ones(len(tickers)))
        weights /= np.sum(weights)

        # Calculate portfolio return
        port_return = np.dot(weights, returns_list)
        portfolio_returns.append(round(port_return, 4))

        # Calculate portfolio volatility
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)) * 12)
        portfolio_volatility.append(round(port_volatility, 4))

        # Sharpe ratio assuming rf=2%
        sharpe_ratio = (port_return - rf) / port_volatility if port_volatility != 0 else 0
        portfolio_sharpe.append(round(sharpe_ratio, 4))

        portfolio_weights.append(np.round(weights, 4))

    # Create portfolios DataFrame
    portfolios_df = pd.DataFrame({
        'Portfolio Return': portfolio_returns,
        'Portfolio Risk': portfolio_volatility,
        'Sharpe Ratio': portfolio_sharpe,
        'Weights by Stock': portfolio_weights
    })

    return portfolios_df, covariance_matrix, returns_list

# Use the function above to generate portfolios for good and poor ESG companies
good_esg_portfolios, good_esg_cov_matrix, good_esg_returns = generate_portfolios(good_esg_tickers)
poor_esg_portfolios, poor_esg_cov_matrix, poor_esg_returns = generate_portfolios(poor_esg_tickers)

# Output generated portfolios
print("\n[Good ESG Portfolios]")
print(good_esg_portfolios)

print("\n[Poor ESG Portfolios]")
print(poor_esg_portfolios)

# Plotting generated portfolios
plt.figure(figsize=(14, 6))

# Good ESG Portfolio Performance
plt.subplot(1, 2, 1)
plt.scatter(good_esg_portfolios['Portfolio Risk'], good_esg_portfolios['Portfolio Return'], 
            c=good_esg_portfolios['Sharpe Ratio'], cmap='viridis', marker='.', alpha=0.7, label='Good ESG Portfolios')
plt.colorbar(label='Sharpe Ratio')
plt.title('Good ESG Portfolios')
plt.xlabel('Portfolio Risk (Volatility)')
plt.ylabel('Portfolio Return')
plt.legend(loc='best')
plt.grid()

# Poor ESG Portfolio Performance
plt.subplot(1, 2, 2)
plt.scatter(poor_esg_portfolios['Portfolio Risk'], poor_esg_portfolios['Portfolio Return'], 
            c=poor_esg_portfolios['Sharpe Ratio'], cmap='plasma', marker='.', alpha=0.7, label='Poor ESG Portfolios')
plt.colorbar(label='Sharpe Ratio')
plt.title('Poor ESG Portfolios')
plt.xlabel('Portfolio Risk (Volatility)')
plt.ylabel('Portfolio Return')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()

# 4
# Locate the global minimum variance portfolio and the optimal risky portfolio

# Function to find GMVP and ORP
def find_gmvp_and_orp(portfolios_df):
    # Find Optimal Risky Portfolio (ORP) - max Sharpe ratio
    opt_port = portfolios_df.iloc[portfolios_df['Sharpe Ratio'].idxmax()]

    # Find Global Minimum Variance Portfolio (GMVP) - min volatility
    min_var_port = portfolios_df.iloc[portfolios_df['Portfolio Risk'].idxmin()]

    return opt_port, min_var_port

# Use the function above to find GMVP and ORP for good and poor ESG portfolios
good_esg_opt_port, good_esg_min_var_port = find_gmvp_and_orp(good_esg_portfolios)
poor_esg_opt_port, poor_esg_min_var_port = find_gmvp_and_orp(poor_esg_portfolios)

# Output expected returns, standard deviation, weights, and Sharpe ratio for GMVP and ORP
print("[Good ESG Portfolios]")
print(f"Optimal Risky Portfolio (ORP):\n{good_esg_opt_port}\n")
print(f"Global Minimum Variance Portfolio (GMVP):\n{good_esg_min_var_port}\n")

print("[Poor ESG Portfolios]")
print(f"Optimal Risky Portfolio (ORP):\n{poor_esg_opt_port}\n")
print(f"Global Minimum Variance Portfolio (GMVP):\n{poor_esg_min_var_port}\n")

# Plotting Portfolio Performance with GMVP and ORP
plt.figure(figsize=(14, 6))

# Good ESG Portfolio Performance
plt.subplot(1, 2, 1)

plt.scatter(good_esg_portfolios['Portfolio Risk'], good_esg_portfolios['Portfolio Return'], 
            c=good_esg_portfolios['Sharpe Ratio'], cmap='viridis', marker='.', alpha=0.7)
plt.scatter(good_esg_opt_port['Portfolio Risk'], good_esg_opt_port['Portfolio Return'], 
            color='red', marker='*', s=200, label='Optimal Risky Portfolio')
plt.scatter(good_esg_min_var_port['Portfolio Risk'], good_esg_min_var_port['Portfolio Return'], 
            color='blue', marker='*', s=200, label='Global Minimum Variance Portfolio')
plt.colorbar(label='Sharpe Ratio')
plt.title('Good ESG Portfolios with GMVP and ORP')
plt.xlabel('Portfolio Risk (Volatility)')
plt.ylabel('Portfolio Return')
plt.legend(loc='best')
plt.grid()

# Poor ESG Portfolio Performance
plt.subplot(1, 2, 2)
plt.scatter(poor_esg_portfolios['Portfolio Risk'], poor_esg_portfolios['Portfolio Return'], 
            c=poor_esg_portfolios['Sharpe Ratio'], cmap='plasma', marker='.', alpha=0.7)
plt.scatter(poor_esg_opt_port['Portfolio Risk'], poor_esg_opt_port['Portfolio Return'], 
            color='red', marker='*', s=200, label='Optimal Risky Portfolio')
plt.scatter(poor_esg_min_var_port['Portfolio Risk'], poor_esg_min_var_port['Portfolio Return'], 
            color='blue', marker='*', s=200, label='Global Minimum Variance Portfolio')
plt.colorbar(label='Sharpe Ratio')
plt.title('Poor ESG Portfolios with GMVP and ORP')
plt.xlabel('Portfolio Risk (Volatility)')
plt.ylabel('Portfolio Return')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()

# 5
# Draw minimum variance frontiers for two groups

# Function to calculate the Minimum Variance Frontier (MVF)
def calculate_mvf(cov_matrix, returns_list, num_points=100):

    # Initialize lists to store MVF metrics
    weights = []
    returns = []
    risks = []

    # Find the minimum and maximum returns
    min_return, max_return = min(returns_list), max(returns_list)
    target_returns = np.linspace(min_return, max_return, num_points)

    # Calculate the MVF
    for target_return in target_returns:
        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))*12)# Annualized

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: np.dot(x, returns_list) - target_return})

        bounds = tuple((0, 1) for _ in range(len(returns_list)))
        result = minimize(portfolio_risk, len(returns_list) * [1. / len(returns_list)], bounds=bounds, constraints=constraints)

        if result.success:
            weights.append(result.x)
            returns.append(target_return)
            risks.append(result.fun)

    return np.array(returns), np.array(risks), np.array(weights)

# Use the function above to calculate MVF for good and poor ESG portfolios
good_esg_mvf_returns, good_esg_mvf_risks, good_esg_mvf_weights = calculate_mvf(good_esg_cov_matrix, [calculate_metrics(ticker)[1] for ticker in good_esg_tickers])
poor_esg_mvf_returns, poor_esg_mvf_risks, poor_esg_mvf_weights = calculate_mvf(poor_esg_cov_matrix, [calculate_metrics(ticker)[1] for ticker in poor_esg_tickers])

# Plotting Minimum Variance Frontiers
plt.figure(figsize=(14, 6))

# Good ESG Group
plt.subplot(1, 2, 1)
plt.scatter(good_esg_portfolios["Portfolio Risk"], good_esg_portfolios["Portfolio Return"], c=(good_esg_portfolios["Portfolio Return"]-0.02) / good_esg_portfolios["Portfolio Risk"], cmap='viridis', marker='.', label='Good ESG Portfolios')
plt.colorbar(label='Sharpe Ratio')
plt.plot(good_esg_mvf_risks, good_esg_mvf_returns, 'r--', linewidth=1, label='Good ESG MVF')
plt.scatter(good_esg_opt_port['Portfolio Risk'], good_esg_opt_port['Portfolio Return'], color='red', marker='*', s=200, label="Good ESG ORP")
plt.title('Good ESG Portfolios & MVF')
plt.xlabel('Portfolio Risk (Volatility)')
plt.ylabel('Portfolio Return')
plt.legend(loc='best')
plt.grid()

# Poor ESG Group
plt.subplot(1, 2, 2)
plt.scatter(poor_esg_portfolios["Portfolio Risk"], poor_esg_portfolios["Portfolio Return"], c=(poor_esg_portfolios["Portfolio Return"]-0.02) / poor_esg_portfolios["Portfolio Risk"], cmap='plasma', marker='.', label='Poor ESG Portfolios')
plt.colorbar(label='Sharpe Ratio')
plt.plot(poor_esg_mvf_risks, poor_esg_mvf_returns, 'r--', linewidth=1, label='Poor ESG MVF')
plt.scatter(poor_esg_opt_port['Portfolio Risk'], poor_esg_opt_port['Portfolio Return'], color='red', marker='*', s=200, label="Poor ESG ORP")
plt.title('Poor ESG Portfolios & MVF')
plt.xlabel('Portfolio Risk (Volatility)')
plt.ylabel('Portfolio Return')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()

# 6
# Plot CAL and indifference curve(utility function) tangent to mean-variance frontier using P and rf ( = 2 %) for two groups Assume risk aversion = 4
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Risk-free rate
rf = 0.02

# Function to calculate the Capital Allocation Line (CAL)
def cal(optimal_risky_port, rf):

    # Calculate the slope and intercept of the CAL
    cal_x = np.linspace(0, optimal_risky_port['Portfolio Risk'] * 1.5, 1000)
    cal_slope = (optimal_risky_port['Portfolio Return'] - rf) / optimal_risky_port['Portfolio Risk']
    cal_y = rf + cal_slope * cal_x
    return cal_x, cal_y

# Use the function above to calculate CAL for good and poor ESG portfolios
good_cal_x, good_cal_y = cal(good_esg_opt_port, rf)
poor_cal_x, poor_cal_y = cal(poor_esg_opt_port, rf)

# Function to calculate the indifference curve
def indifference_curve(optimal_risky_port, rf, A):

    # Return and volatility of the Optimal Risky Portfolio
    Rp = optimal_risky_port['Portfolio Return']
    sigma_p = optimal_risky_port['Portfolio Risk']

    # Calculate the investment proportion y
    y = (Rp - rf) / (A * sigma_p**2)
    Rc = y * Rp + (1 - y) * rf
    sigma_c = y * sigma_p

    # The utility function
    U = Rc - 0.5 * A * sigma_c**2

    # Calculate the indifference curve
    indifference_sigma = np.linspace(0, sigma_p * 1.5, 100)
    indifference_curve = U + 0.5 * A * indifference_sigma**2
    return indifference_sigma, indifference_curve

# Use the function above to calculate the indifference curve for good and poor ESG portfolios
good_indifference_sigma, good_indifference_curve = indifference_curve(good_esg_opt_port, rf, 4)
poor_indifference_sigma, poor_indifference_curve = indifference_curve(poor_esg_opt_port, rf, 4)

# Plotting Good ESG Group
plt.figure(figsize=(14, 6))

# Good ESG Group
plt.subplot(1, 2, 1)
good_esg_portfolios.plot.scatter(x='Portfolio Risk', y='Portfolio Return', c='Sharpe Ratio', s=2, cmap='viridis', grid=True, colorbar=True, ax=plt.gca())
plt.plot(good_esg_mvf_risks, good_esg_mvf_returns, 'r--', linewidth=1, label='Good ESG MVF')
plt.scatter(good_esg_opt_port['Portfolio Risk'], good_esg_opt_port['Portfolio Return'], color='red', marker='*', s=200, label='Optimal Risky Portfolio')
plt.scatter(good_esg_min_var_port['Portfolio Risk'], good_esg_min_var_port['Portfolio Return'], color='blue', marker='*', s=200, label='Global Minimum Variance Portfolio')
plt.plot(good_cal_x, good_cal_y, label="Capital Allocation Line (CAL)", color='#e45f2b', linewidth=2)
plt.plot(good_indifference_sigma, good_indifference_curve, label="Indifference Curve", linestyle="--", color="#9ac1f0", linewidth=2)
plt.title("Good ESG Group: Efficient Frontier with CAL and Indifference Curve")
plt.xlabel("Volatility")
plt.ylabel("Expected Returns")
plt.legend()
plt.grid(True)

# Poor ESG Group
plt.subplot(1, 2, 2)
poor_esg_portfolios.plot.scatter(x='Portfolio Risk', y='Portfolio Return', c='Sharpe Ratio', s=2, cmap='plasma', grid=True, colorbar=True, ax=plt.gca())
plt.plot(poor_esg_mvf_risks, poor_esg_mvf_returns, 'r--', linewidth=1, label='Poor ESG MVF')
plt.scatter(poor_esg_opt_port['Portfolio Risk'], poor_esg_opt_port['Portfolio Return'], color='red', marker='*', s=200, label='Optimal Risky Portfolio')
plt.scatter(poor_esg_min_var_port['Portfolio Risk'], poor_esg_min_var_port['Portfolio Return'], color='blue', marker='*', s=200, label='Global Minimum Variance Portfolio')
plt.plot(poor_cal_x, poor_cal_y, label="Capital Allocation Line (CAL)", color='#e45f2b', linewidth=2)
plt.plot(poor_indifference_sigma, poor_indifference_curve, label="Indifference Curve", linestyle="--", color="#9ac1f0", linewidth=2)
plt.title("Poor ESG Group: Efficient Frontier with CAL and Indifference Curve")
plt.xlabel("Volatility")
plt.ylabel("Expected Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7
# Construct complete portfolios for two groups

rf = 0.02  # Risk-free rate
A = 4  # Risk aversion coefficient

# Return and volatility of the Good Optimal Risky Portfolio
good_Rp = good_esg_opt_port['Portfolio Return']
good_sigma_p = good_esg_opt_port['Portfolio Risk']

# Return and volatility of the Poor Optimal Risky Portfolio
poor_Rp = poor_esg_opt_port['Portfolio Return']
poor_sigma_p = poor_esg_opt_port['Portfolio Risk']

# Calculate the investment proportion y
y_good = (good_Rp - rf) / (A * good_sigma_p**2)
y_poor = (poor_Rp - rf) / (A * poor_sigma_p**2)

# Return and volatility of the Good Complete Portfolio
Rc_good = y_good * good_Rp + (1 - y_good) * rf
good_sigma_c = y_good * good_sigma_p

# Return and volatility of the Poor Complete Portfolio
Rc_poor = y_poor * poor_Rp + (1 - y_poor) * rf
poor_sigma_c = y_poor * poor_sigma_p

# Tangency condition
U_good = Rc_good - 0.5 * A * good_sigma_c**2
U_poor = Rc_poor - 0.5 * A * poor_sigma_c**2

# Print results
print(f"Good Complete Optimal Portfolio Return (Rc): {Rc_good:.2%}")
print(f"Good Complete Optimal Portfolio Volatility (good_sigma_c): {good_sigma_c:.2%}")
print(f"Proportion Invested in Risky Portfolio (y): {y_good:.2%}")

print(f"Poor Complete Optimal Portfolio Return (Rc): {Rc_poor:.2%}")
print(f"Poor Complete Optimal Portfolio Volatility (poor_sigma_c): {poor_sigma_c:.2%}")
print(f"Proportion Invested in Risky Portfolio (y): {y_poor:.2%}")


plt.figure(figsize=(14, 6))

# Good ESG Group
plt.subplot(1, 2, 1)
good_esg_portfolios.plot.scatter(x='Portfolio Risk', y='Portfolio Return', c='Sharpe Ratio', s=2, cmap='viridis', grid=True, colorbar=True, ax=plt.gca())
plt.plot(good_esg_mvf_risks, good_esg_mvf_returns, 'r--', linewidth=1, label='Good ESG MVF')
plt.plot(good_cal_x, good_cal_y, label="Capital Allocation Line (CAL)", color='#e45f2b', linewidth=2)
plt.plot(good_indifference_sigma, good_indifference_curve, label="Indifference Curve", linestyle="--", color="#9ac1f0", linewidth=2)
plt.scatter(good_esg_opt_port['Portfolio Risk'], good_esg_opt_port['Portfolio Return'], color='red', marker='*', s=200, label='Optimal Risky Portfolio', zorder=5)
plt.scatter(good_esg_min_var_port['Portfolio Risk'], good_esg_min_var_port['Portfolio Return'], color='blue', marker='*', s=200, label='Global Minimum Variance Portfolio', zorder=5)
plt.scatter(good_sigma_c, Rc_good, color='#f6c445', marker='*', s=200, label='Complete Optimal Portfolio', edgecolors='black', zorder=6)
plt.title("Good ESG Group: Efficient Frontier with CAL and Indifference Curve")
plt.xlabel("Volatility")
plt.ylabel("Expected Returns")
plt.legend()
plt.grid(True)

# Poor ESG Group
plt.subplot(1, 2, 2)
poor_esg_portfolios.plot.scatter(x='Portfolio Risk', y='Portfolio Return', c='Sharpe Ratio', s=2, cmap='plasma', grid=True, colorbar=True, ax=plt.gca())
plt.plot(poor_esg_mvf_risks, poor_esg_mvf_returns, 'r--', linewidth=1, label='Poor ESG MVF')
plt.plot(poor_cal_x, poor_cal_y, label="Capital Allocation Line (CAL)", color='#e45f2b', linewidth=2)
plt.plot(poor_indifference_sigma, poor_indifference_curve, label="Indifference Curve", linestyle="--", color="#9ac1f0", linewidth=2)
plt.scatter(poor_esg_opt_port['Portfolio Risk'], poor_esg_opt_port['Portfolio Return'], color='red', marker='*', s=200, label='Optimal Risky Portfolio', zorder=5)
plt.scatter(poor_esg_min_var_port['Portfolio Risk'], poor_esg_min_var_port['Portfolio Return'], color='blue', marker='*', s=200, label='Global Minimum Variance Portfolio', zorder=5)
plt.scatter(poor_sigma_c, Rc_poor, color='#f6c445', marker='*', s=200, label='Complete Optimal Portfolio', edgecolors='black', zorder=6)
plt.title("Poor ESG Group: Efficient Frontier with CAL and Indifference Curve")
plt.xlabel("Volatility")
plt.ylabel("Expected Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8
# Plot the three lines for the two groups int the same graph
plt.figure(figsize=(10, 6))

# Good ESG Group
good_esg_portfolios.plot.scatter(x='Portfolio Risk', y='Portfolio Return', c='Sharpe Ratio', s=2, cmap='viridis', grid=True, colorbar=True, ax=plt.gca())
plt.plot(good_esg_mvf_risks, good_esg_mvf_returns, 'r--', linewidth=1, label='Good ESG MVF')
plt.plot(good_cal_x, good_cal_y, label="Good ESG Capital Allocation Line (CAL)", color='#a0e548', linewidth=2)
plt.plot(good_indifference_sigma, good_indifference_curve, label="Good ESG Indifference Curve", linestyle="--", color="#9ac1f0", linewidth=2)
plt.scatter(good_esg_opt_port['Portfolio Risk'], good_esg_opt_port['Portfolio Return'], color='blue', marker='*', s=200, label='Good ESG Optimal Risky Portfolio', zorder=5)
plt.scatter(good_sigma_c, Rc_good, color='#f6c445', marker='*', s=200, label='Good ESG Complete Optimal Portfolio', edgecolors='black', zorder=6)

# Poor ESG Group
poor_esg_portfolios.plot.scatter(x='Portfolio Risk', y='Portfolio Return', c='Sharpe Ratio', s=2, cmap='plasma', grid=True, colorbar=True, ax=plt.gca())
plt.plot(poor_esg_mvf_risks, poor_esg_mvf_returns, 'r--', linewidth=1, label='Poor ESG MVF')
plt.plot(poor_cal_x, poor_cal_y, label="Poor ESG Capital Allocation Line (CAL)", color='#e45f2b', linewidth=2)
plt.plot(poor_indifference_sigma, poor_indifference_curve, label="Poor ESG Indifference Curve", linestyle="--", color="#f6c445", linewidth=2)
plt.scatter(poor_esg_opt_port['Portfolio Risk'], poor_esg_opt_port['Portfolio Return'], color='red', marker='*', s=200, label='Poor ESG Optimal Risky Portfolio', zorder=5)
plt.scatter(poor_sigma_c, Rc_poor, color='#fbd0e0', marker='*', s=200, label='Poor ESG Complete Optimal Portfolio', edgecolors='black', zorder=6)

plt.title("Efficient Frontier with CAL and Indifference Curve for Good and Poor ESG Groups")
plt.xlabel("Volatility")
plt.ylabel("Expected Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

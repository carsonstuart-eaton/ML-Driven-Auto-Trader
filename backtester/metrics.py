def annualized_return(cumulative_return, periods_per_year=252):
    return cumulative_return ** (periods_per_year / len(cumulative_return)) - 1

def annualized_volatility(daily_returns, periods_per_year=252):
    return daily_returns.std() * (periods_per_year ** 0.5)

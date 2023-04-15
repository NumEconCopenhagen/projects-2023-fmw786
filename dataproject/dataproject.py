class eeff:

    def hello_world():
        print("Hello, world!")

    def down():
        #Choose 20 companies
        tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "CLX", "CMA", "CTRA", "ELV", "ESS", "EXR", "FANG", "FMC", "FTV", "GOOGL", "GS", "HAL", "IVZ", "JPM", "NWL", "NFLX", ]

        #Dowload data for the last 10 years
        start_date = "2013-01-01"
        end_date = "2023-01-01"

        ohlc = yf.download(tickers, start=start_date, end=end_date)
        prices = ohlc["Adj Close"].dropna(how="all")
        prices.tail()

        #Plot data
        prices[prices.index >= "2008-01-01"].plot(figsize=(15,8));

    def cov():
        #Import
        import pypfopt
        pypfopt.__version__

        #Covariance
        from pypfopt import risk_models
        from pypfopt import plotting

        sample_cov = risk_models.sample_cov(prices, frequency=252)
        sample_cov

        #Plot covariance
        plotting.plot_covariance(sample_cov, plot_correlation=True);

    def covLW():
        #Ledoit-Wolf shrinkage (reduces extreme values in the covariance matrix)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        plotting.plot_covariance(S, plot_correlation=True);

    def expected_returns():
        #Calculate expected returns
        from pypfopt import expected_returns

        mu = expected_returns.capm_return(prices)
        mu

        #Plot:
        mu.plot.barh(figsize=(10,6));

    def efficientfrontier():
        #Import
        from pypfopt import EfficientFrontier

        #Give weights
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        # You don't have to provide expected returns in this case
        ef = EfficientFrontier(None, S, weight_bounds=(None, None))
        ef.min_volatility()
        weights = ef.clean_weights()
        weights

        #Plot
        pd.Series(weights).plot.barh();

        #Performance
        ef.portfolio_performance(verbose=True);

        #Optimize
        from pypfopt import CLA, plotting

        cla = CLA(mu, S)
        cla.max_sharpe()
        cla.portfolio_performance(verbose=True);

        #Plot
        ax = plotting.plot_efficient_frontier(cla, showfig=False)

    def random_portefolios():
        #Create 1.000.000 random portefolios
        n_samples = 1000000
        w = np.random.dirichlet(np.ones(len(mu)), n_samples)
        rets = w.dot(mu)
        stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
        sharpes = rets / stds

        print("Sample portfolio returns:", rets)
        print("Sample portfolio volatilities:", stds)

        #Plot the efficient frontier against the 1.000.000 random
        ef = EfficientFrontier(mu, S)

        fig, ax = plt.subplots()
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

        # Find and plot the tangency portfolio
        ef2 = EfficientFrontier(mu, S)
        ef2.max_sharpe()
        ret_tangent, std_tangent, _ = ef2.portfolio_performance()

        # Plot random portefolios
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

        # Format
        ax.set_title("Efficient Frontier and Random Portfolios")
        ax.legend()
        plt.tight_layout()
        plt.show()

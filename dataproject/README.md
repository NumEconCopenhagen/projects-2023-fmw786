# Data analysis project - Portefolio Optimization in a CAPM framework

Our project is titled "Portefolio Optimization in a CAPM framework" and seeks to put together an optimal portofolio for 20 companies over the last 10 years in a CAPM framework. We seek to compute, plot and present the following:

  1) Prices for the 20 companies over the last 10 years
  2) The standard covariance matrix (20x20) and the covariance matrix (20x20) with a Ledoit-Wolf shrinkage method 
  3) The expected returns
  4) The efficient frontier (along with 1.000.000 random put together portofolios in order to show how they all lay under the efficient frontier)
  5) The sharpe ratio, expected return and volatolity of the optimal portefolio in a simple CAPM-model

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).

Our data is imported directly from Yahoo Finance with and API and we therefore do not have a .CSV-file or something like that. 

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install matplotlib-venn``
``pip install yahoo_fin``
``pip install PyPortfolioOpt``

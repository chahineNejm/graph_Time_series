# single keyword, see candidates first
python scrape_ts_notebooks.py -k "fourier transform forecasting" --dry-run

# then pull
python scrape_ts_notebooks.py -k "wavelet time series" --max-per-query 30
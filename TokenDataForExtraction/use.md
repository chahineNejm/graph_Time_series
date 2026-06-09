# single keyword, see candidates first
python TokenDataForExtraction\scrape_kaggle.py -k "fourier transform forecasting" --dry-run

# then pull
python scrape_ts_notebooks.py -k "wavelet time series" --max-per-query 30
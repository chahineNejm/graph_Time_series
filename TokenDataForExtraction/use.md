# Kaggle Notebook Scraping Notes

Use this folder for exploratory notebook scraping near the scraping scripts.
Keep API keys out of this directory and out of git.

Preview candidates first:

```bash
python TokenDataForExtraction\scrape_kaggle.py -k "fourier transform forecasting" --dry-run
```

Then pull a bounded set:

```bash
python TokenDataForExtraction\scrape_kaggle.py -k "wavelet time series" --max-per-query 30
```

Scraped data should stay close to the scraping code, not in a distant parent
folder.

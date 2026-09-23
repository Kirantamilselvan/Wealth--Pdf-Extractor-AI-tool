"""
Pull SEC filings (10-K / 10-Q / 8-K) for a list of tickers from SEC EDGAR.

SEC requires a descriptive User-Agent header on every request (name + email).
Set EDGAR_USER_AGENT in your .env file.

Usage:
    python ingestion/edgar_fetch.py --tickers AAPL MSFT JPM --forms 10-K --limit 5
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.getenv("EDGAR_USER_AGENT", "Wealth Doc Intelligence contact@example.com")
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
HEADERS = {"User-Agent": USER_AGENT}

TICKER_TO_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"


def load_ticker_map() -> dict:
    resp = requests.get(TICKER_TO_CIK_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def fetch_filings_for_cik(cik: str, forms: list[str], limit: int) -> list[dict]:
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    recent = data["filings"]["recent"]

    results = []
    for i, form in enumerate(recent["form"]):
        if form in forms:
            results.append(
                {
                    "cik": cik,
                    "form": form,
                    "accession_number": recent["accessionNumber"][i],
                    "filing_date": recent["filingDate"][i],
                    "primary_document": recent["primaryDocument"][i],
                    "company_name": data.get("name", ""),
                }
            )
        if len(results) >= limit:
            break
    return results


def download_filing(filing: dict, ticker: str) -> Path:
    cik_int = int(filing["cik"])
    accession_nodash = filing["accession_number"].replace("-", "")
    doc_url = ARCHIVE_URL.format(
        cik_int=cik_int, accession_nodash=accession_nodash, doc=filing["primary_document"]
    )
    resp = requests.get(doc_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    out_dir = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filing['form']}_{filing['filing_date']}.html"
    out_path.write_bytes(resp.content)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(filing, indent=2))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT JPM")
    parser.add_argument("--forms", nargs="+", default=["10-K"], help="e.g. 10-K 10-Q 8-K")
    parser.add_argument("--limit", type=int, default=3, help="filings per ticker")
    args = parser.parse_args()

    print("Loading ticker -> CIK map...")
    ticker_map = load_ticker_map()

    for ticker in args.tickers:
        ticker = ticker.upper()
        cik = ticker_map.get(ticker)
        if not cik:
            print(f"  [skip] {ticker}: not found in EDGAR ticker map")
            continue

        print(f"  Fetching {args.forms} for {ticker} (CIK {cik})...")
        filings = fetch_filings_for_cik(cik, args.forms, args.limit)
        for filing in filings:
            path = download_filing(filing, ticker)
            print(f"    saved {path}")
            time.sleep(0.2)  # be polite to SEC rate limits (10 req/sec max)

    print("Done.")


if __name__ == "__main__":
    main()

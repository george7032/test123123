# Reconciliation Studio

Streamlit app that reconciles two datasets (bank vs ledger, sales vs payments, etc.).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How matching works
Each row is matched at most once, in this order:
1. **Reference + amount**: same cleaned reference, amount within tolerance.
2. **Amount + date**: amount within tolerance and date within N days (for rows with no/different reference).
3. **Reference match, amount differs**: reported separately as an amount mismatch.

Whatever is left is **Only in A** / **Only in B**. A "reconciliation proof" table shows that
(Total A − Total B) is fully explained by the categories above (unexplained should be 0.00).

## Files
- `app.py`: Streamlit interface
- `recon.py`: matching engine (usable without Streamlit)

import io

import pandas as pd
import streamlit as st

import recon

st.set_page_config(page_title="Reconciliation Studio", page_icon="🔁", layout="wide")


# ----------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def sheet_names(data: bytes):
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner=False)
def load_file(data: bytes, filename: str, sheet):
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0, dtype=object)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=object, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file as CSV.")


def guess(cols, words, default=None):
    for w in words:
        for c in cols:
            if w in str(c).lower():
                return c
    return default


def uploader(label, key):
    up = st.file_uploader(f"{label} (CSV or Excel)", type=["csv", "xlsx", "xls", "xlsm"], key=f"up_{key}")
    if not up:
        return None
    data = up.getvalue()
    sheet = None
    if up.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        names = sheet_names(data)
        if len(names) > 1:
            sheet = st.selectbox("Sheet", names, key=f"sheet_{key}")
    try:
        return load_file(data, up.name, sheet)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read file: {e}")
        return None


def mapping_ui(df: pd.DataFrame, key: str) -> recon.Mapping:
    cols = list(df.columns)
    mode = st.radio("Amount layout", ["Single amount column", "Debit & Credit columns"],
                    horizontal=True, key=f"mode_{key}")
    m = recon.Mapping()
    if mode == "Single amount column":
        i = cols.index(guess(cols, ["amount", "value", "total"], cols[0]))
        m.amount_col = st.selectbox("Amount column", cols, index=i, key=f"amt_{key}")
    else:
        none = ["(none)"] + cols
        d = st.selectbox("Debit column", none, index=none.index(guess(cols, ["debit", "dr", "withdraw"], "(none)")), key=f"dr_{key}")
        c = st.selectbox("Credit column", none, index=none.index(guess(cols, ["credit", "cr", "deposit"], "(none)")), key=f"cr_{key}")
        m.debit_col = None if d == "(none)" else d
        m.credit_col = None if c == "(none)" else c
    default_ref = guess(cols, ["ref", "invoice", "doc", "txn", "transaction", "id"])
    m.ref_cols = st.multiselect("Reference / ID column(s) — optional", cols,
                                default=[default_ref] if default_ref else [], key=f"ref_{key}",
                                help="Several columns are combined into one key.")
    none = ["(none)"] + cols
    dcol = st.selectbox("Date column — optional", none,
                        index=none.index(guess(cols, ["date"], "(none)")), key=f"date_{key}")
    m.date_col = None if dcol == "(none)" else dcol
    m.invert_sign = st.checkbox("Flip sign of amounts", key=f"inv_{key}",
                                help="Use when one system shows payments as negatives and the other as positives.")
    return m


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def money(x):
    return f"{x:,.2f}"


# ---------------------------------------------------------------------- UI
st.title("🔁 Reconciliation Studio")
st.caption("Upload two datasets (e.g. bank statement vs ledger), map the columns, and see what matches, what doesn't, and why.")

with st.sidebar:
    st.header("Settings")
    name_a = st.text_input("Name of dataset A", "Bank")
    name_b = st.text_input("Name of dataset B", "Ledger")
    st.subheader("Matching rules")
    tol = st.number_input("Amount tolerance (±)", min_value=0.0, value=0.01, step=0.01, format="%.2f",
                          help="Amounts differing by no more than this count as equal.")
    window = st.number_input("Date window (± days)", min_value=0, value=3, step=1)
    req_date = st.checkbox("Require date window on reference matches", value=False)
    pass2 = st.checkbox("Also match on amount + date when reference is missing/different", value=True)
    amount_only = st.checkbox("Allow amount-only matching if there are no dates", value=False,
                              help="Weak: identical amounts can be paired up wrongly. Review results.")
    flag_mm = st.checkbox("Flag 'same reference, different amount' as mismatches", value=True)
    st.subheader("Parsing")
    dayfirst = st.checkbox("Dates are day-first (31/12/2026)", value=True)
    dec_comma = st.checkbox("Decimal comma (1.234,56)", value=False)
    strip0 = st.checkbox("Ignore leading zeros in references", value=True)
    st.divider()
    use_sample = st.checkbox("Use built-in sample data", value=False)

rules = recon.Rules(amount_tol=tol, day_window=int(window), require_date_on_ref=req_date,
                    match_amount_date=pass2, amount_only_if_no_dates=amount_only,
                    flag_mismatches=flag_mm, dayfirst=dayfirst, decimal_comma=dec_comma,
                    strip_zeros=strip0)

# ---- Step 1: upload
st.subheader("1 · Upload")
c1, c2 = st.columns(2)
if use_sample:
    df_a, df_b = recon.make_sample()
    c1.info("Sample bank data loaded")
    c2.info("Sample ledger data loaded")
else:
    with c1:
        df_a = uploader(name_a, "a")
    with c2:
        df_b = uploader(name_b, "b")

if df_a is None or df_b is None:
    st.info("Upload both files (or tick **Use built-in sample data** in the sidebar) to continue.")
    st.stop()

with st.expander("Preview data"):
    p1, p2 = st.columns(2)
    p1.write(f"**{name_a}** — {len(df_a):,} rows")
    p1.dataframe(df_a.head(50), use_container_width=True)
    p2.write(f"**{name_b}** — {len(df_b):,} rows")
    p2.dataframe(df_b.head(50), use_container_width=True)

# ---- Step 2: mapping
st.subheader("2 · Map columns")
m1, m2 = st.columns(2)
with m1:
    st.markdown(f"**{name_a}**")
    map_a = mapping_ui(df_a, "a")
with m2:
    st.markdown(f"**{name_b}**")
    map_b = mapping_ui(df_b, "b")

# ---- Step 3: run
st.subheader("3 · Reconcile")
if st.button("Run reconciliation", type="primary"):
    if not (map_a.amount_col or map_a.debit_col or map_a.credit_col) or not (map_b.amount_col or map_b.debit_col or map_b.credit_col):
        st.error("Choose an amount column (or debit/credit) for both files.")
    else:
        with st.spinner("Matching…"):
            A = recon.prepare(df_a, map_a, rules)
            B = recon.prepare(df_b, map_b, rules)
            st.session_state["res"] = recon.reconcile(A, B, rules)
            st.session_state["names"] = (name_a, name_b)

res = st.session_state.get("res")
if not res:
    st.stop()
name_a, name_b = st.session_state["names"]
s = res["summary"]

# ---- Results
st.divider()
k = st.columns(5)
k[0].metric("Matched pairs", f"{s['matched']:,}")
k[1].metric("Amount mismatches", f"{s['mismatched']:,}")
k[2].metric(f"Only in {name_a}", f"{s['only_a']:,}")
k[3].metric(f"Only in {name_b}", f"{s['only_b']:,}")
k[4].metric("Match rate", f"{s['match_rate_a']:.1%} / {s['match_rate_b']:.1%}", help=f"{name_a} / {name_b}")

warn = []
if s["bad_amount_a"] or s["bad_amount_b"]:
    warn.append(f"Unreadable amounts: {s['bad_amount_a']} in {name_a}, {s['bad_amount_b']} in {name_b} (cannot be matched).")
if s["bad_date_a"] or s["bad_date_b"]:
    warn.append(f"Unreadable dates: {s['bad_date_a']} in {name_a}, {s['bad_date_b']} in {name_b}.")
if len(res["dupes_a"]) or len(res["dupes_b"]):
    warn.append(f"Possible duplicates: {len(res['dupes_a'])} rows in {name_a}, {len(res['dupes_b'])} in {name_b}.")
for w in warn:
    st.warning(w)

with st.expander("Why do the totals differ? (reconciliation proof)", expanded=True):
    proof = pd.DataFrame([
        (f"Total {name_a}", s["sum_a"]),
        (f"Total {name_b}", s["sum_b"]),
        ("Gap (A − B)", s["gap"]),
        ("   Differences inside tolerance (matched pairs)", s["matched_diff"]),
        ("   Amount mismatches", s["mismatch_diff"]),
        (f"   Only in {name_a}", s["only_a_total"]),
        (f"   Only in {name_b} (subtracted)", -s["only_b_total"]),
        ("   Unexplained (should be 0.00)", s["unexplained"]),
    ], columns=["Item", "Amount"])
    proof["Amount"] = proof["Amount"].map(money)
    st.dataframe(proof, hide_index=True, use_container_width=True)
    if s["by_type"]:
        st.caption("Matches by method: " + ", ".join(f"{k} = {v}" for k, v in s["by_type"].items()))

st.download_button("⬇️ Download full report (Excel)", recon.to_excel(res, name_a, name_b),
                   file_name="reconciliation_report.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

tabs = st.tabs([f"Matched ({s['matched']})", f"Amount mismatches ({s['mismatched']})",
                f"Only in {name_a} ({s['only_a']})", f"Only in {name_b} ({s['only_b']})",
                "Duplicates"])
frames = [("matched", res["matched"]), ("mismatches", res["mismatches"]),
          ("only_a", res["only_a"]), ("only_b", res["only_b"])]
for tab, (nm, df) in zip(tabs[:4], frames):
    with tab:
        if df.empty:
            st.success("Nothing here 🎉")
            continue
        if nm == "matched":
            kinds = st.multiselect("Filter by match method", sorted(df["match_type"].unique()),
                                   default=sorted(df["match_type"].unique()), key="kinds")
            df = df[df["match_type"].isin(kinds)]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", csv_bytes(df), file_name=f"{nm}.csv", key=f"dl_{nm}")
with tabs[4]:
    for nm, df in ((name_a, res["dupes_a"]), (name_b, res["dupes_b"])):
        st.markdown(f"**{nm}** — {len(df)} rows share the same reference, amount and date")
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

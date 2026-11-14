"""Reconciliation engine: matches two datasets (e.g. bank vs ledger).

Matching passes (each row is used at most once):
  1. Reference + amount (within tolerance, optionally date window)
  2. Amount + date window (no reference needed)
  3. Reference matches but amount differs  -> reported as "Amount mismatch"
Whatever is left is "Only in A" / "Only in B".
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HELPERS = ["src_row", "std_ref", "std_amount", "std_date"]


@dataclass
class Mapping:
    ref_cols: list = field(default_factory=list)
    amount_col: str | None = None
    debit_col: str | None = None
    credit_col: str | None = None
    date_col: str | None = None
    invert_sign: bool = False


@dataclass
class Rules:
    amount_tol: float = 0.0
    day_window: int = 3
    require_date_on_ref: bool = False
    match_amount_date: bool = True
    amount_only_if_no_dates: bool = False
    flag_mismatches: bool = True
    dayfirst: bool = False
    decimal_comma: bool = False
    strip_zeros: bool = True


# ----------------------------------------------------------------- cleaning
def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_))


def parse_amount(s: pd.Series, decimal_comma: bool = False) -> pd.Series:
    """Parse messy money strings: 'KES 1,234.50', '(500)', '100-', '-12.3'."""
    numeric_mask = s.map(_is_num)
    t = s.astype(str).str.strip()
    neg = t.str.match(r"^\(.*\)$") | t.str.contains("-", regex=False)
    t = t.str.replace(r"[^\d.,]", "", regex=True)
    if decimal_comma:
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        t = t.str.replace(",", "", regex=False)
    v = pd.to_numeric(t, errors="coerce").abs().astype(float)
    v = v.where(~neg, -v)
    # genuine numeric cells (e.g. from Excel) are used as-is
    if numeric_mask.any():
        v[numeric_mask] = pd.to_numeric(s[numeric_mask], errors="coerce").astype(float)
    return v


def _clean_ref_value(x) -> str:
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return ""
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        x = int(x)
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def make_ref(df: pd.DataFrame, cols: list, strip_zeros: bool) -> pd.Series:
    if not cols:
        return pd.Series("", index=df.index)
    parts = []
    for c in cols:
        p = df[c].map(_clean_ref_value)
        if strip_zeros:
            p = p.str.lstrip("0")
        parts.append(p)
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    blank = out.str.replace("|", "", regex=False).eq("")
    return out.where(~blank, "")


def parse_date(s: pd.Series, dayfirst: bool) -> pd.Series:
    try:
        d = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
    except (TypeError, ValueError):
        d = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize()


def prepare(df: pd.DataFrame, m: Mapping, rules: Rules) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    for h in HELPERS:
        if h in out.columns:
            out = out.rename(columns={h: f"{h}_orig"})
    out.insert(0, "src_row", out.index + 2)  # row number as seen in Excel (header = 1)

    if m.amount_col:
        amt = parse_amount(out[m.amount_col], rules.decimal_comma)
    else:
        cr = parse_amount(out[m.credit_col], rules.decimal_comma).abs() if m.credit_col else pd.Series(np.nan, index=out.index)
        dr = parse_amount(out[m.debit_col], rules.decimal_comma).abs() if m.debit_col else pd.Series(np.nan, index=out.index)
        amt = cr.fillna(0) - dr.fillna(0)
        amt = amt.where(cr.notna() | dr.notna())
    if m.invert_sign:
        amt = -amt
    out["std_amount"] = amt.round(4)
    out["std_ref"] = make_ref(out, m.ref_cols, rules.strip_zeros)
    if m.date_col:
        out["std_date"] = parse_date(out[m.date_col], rules.dayfirst)
    else:
        out["std_date"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    return out


# ---------------------------------------------------------------- matching
def _slim(df: pd.DataFrame, s: str) -> pd.DataFrame:
    return df[HELPERS].rename(columns={
        "src_row": f"row_{s}", "std_ref": f"ref_{s}",
        "std_amount": f"amt_{s}", "std_date": f"date_{s}"})


def _score(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["amt_diff"] = (c["amt_a"] - c["amt_b"]).round(4)
    c["day_diff"] = (c["date_a"] - c["date_b"]).dt.days.abs()
    return c


def _greedy(c: pd.DataFrame, match_type: str) -> pd.DataFrame:
    c = c.assign(_s1=c["amt_diff"].abs(), _s2=c["day_diff"].fillna(1e6)).sort_values(["_s1", "_s2"], kind="stable")
    used_a, used_b, keep = set(), set(), []
    for i, (a, b) in enumerate(zip(c["row_a"].to_numpy(), c["row_b"].to_numpy())):
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        keep.append(i)
    res = c.iloc[keep][["row_a", "row_b", "amt_diff", "day_diff"]].copy()
    res.insert(0, "match_type", match_type)
    return res


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame({"match_type": pd.Series(dtype="object"),
                         "row_a": pd.Series(dtype="int64"),
                         "row_b": pd.Series(dtype="int64"),
                         "amt_diff": pd.Series(dtype="float64"),
                         "day_diff": pd.Series(dtype="float64")})


def _combine(A, B, pairs, tag_a="A | ", tag_b="B | "):
    if pairs.empty:
        cols = ["match_type", "amt_diff", "day_diff"] + [tag_a + c for c in A.columns] + [tag_b + c for c in B.columns]
        return pd.DataFrame(columns=cols)
    out = (pairs.merge(A.add_prefix(tag_a), left_on="row_a", right_on=tag_a + "src_row")
                .merge(B.add_prefix(tag_b), left_on="row_b", right_on=tag_b + "src_row")
                .drop(columns=["row_a", "row_b"]))
    return out


def _dupes(df: pd.DataFrame) -> pd.DataFrame:
    usable = df["std_amount"].notna() & ((df["std_ref"] != "") | df["std_date"].notna())
    mask = usable & df.duplicated(["std_ref", "std_amount", "std_date"], keep=False)
    return df[mask].sort_values(["std_ref", "std_amount", "std_date"])


def reconcile(A: pd.DataFrame, B: pd.DataFrame, rules: Rules) -> dict:
    tol = float(rules.amount_tol) + 1e-9
    parts = []
    remA, remB = A, B

    # ---- pass 1: reference + amount
    a = _slim(remA, "a"); a = a[a["ref_a"] != ""]
    b = _slim(remB, "b"); b = b[b["ref_b"] != ""]
    c = _score(a.merge(b, left_on="ref_a", right_on="ref_b"))
    ok = c["amt_diff"].abs() <= tol
    if rules.require_date_on_ref:
        ok &= c["day_diff"] <= rules.day_window
    p1 = _greedy(c[ok], "Reference + amount")
    parts.append(p1)
    remA = remA[~remA["src_row"].isin(p1["row_a"])]
    remB = remB[~remB["src_row"].isin(p1["row_b"])]

    # ---- pass 2: amount + date window
    has_dates = A["std_date"].notna().any() and B["std_date"].notna().any()
    if rules.match_amount_date and (has_dates or rules.amount_only_if_no_dates):
        a = _slim(remA, "a").dropna(subset=["amt_a"])
        b = _slim(remB, "b").dropna(subset=["amt_b"])
        w = max(float(rules.amount_tol), 0.005)
        a = a.assign(bk=np.floor(a["amt_a"] / w).astype("int64"))
        b = b.assign(bk=np.floor(b["amt_b"] / w).astype("int64"))
        b3 = pd.concat([b.assign(bk=b["bk"] + s) for s in (-1, 0, 1)])
        c = _score(a.merge(b3, on="bk"))
        ok = c["amt_diff"].abs() <= tol
        if has_dates:
            ok &= c["day_diff"] <= rules.day_window
            label = "Amount + date"
        else:
            label = "Amount only (weak)"
        p2 = _greedy(c[ok], label)
        parts.append(p2)
        remA = remA[~remA["src_row"].isin(p2["row_a"])]
        remB = remB[~remB["src_row"].isin(p2["row_b"])]

    pairs = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else _empty_pairs()
    pairs = pairs.astype({"row_a": "int64", "row_b": "int64"})

    # ---- pass 3: same reference, different amount
    mism = _empty_pairs()
    if rules.flag_mismatches:
        a = _slim(remA, "a"); a = a[a["ref_a"] != ""]
        b = _slim(remB, "b"); b = b[b["ref_b"] != ""]
        c = _score(a.merge(b, left_on="ref_a", right_on="ref_b"))
        c = c.dropna(subset=["amt_diff"])
        if not c.empty:
            mism = _greedy(c, "Reference match, amount differs").astype({"row_a": "int64", "row_b": "int64"})
        remA = remA[~remA["src_row"].isin(mism["row_a"])]
        remB = remB[~remB["src_row"].isin(mism["row_b"])]

    matched = _combine(A, B, pairs)
    mismatches = _combine(A, B, mism)

    # ---- reconciliation proof
    sum_a, sum_b = A["std_amount"].sum(), B["std_amount"].sum()
    md = pairs["amt_diff"].sum()
    xd = mism["amt_diff"].sum()
    un_a, un_b = remA["std_amount"].sum(), remB["std_amount"].sum()
    unexplained = round(sum_a - sum_b - (md + xd + un_a - un_b), 4)

    summary = {
        "rows_a": len(A), "rows_b": len(B),
        "matched": len(pairs), "mismatched": len(mism),
        "only_a": len(remA), "only_b": len(remB),
        "match_rate_a": len(pairs) / len(A) if len(A) else 0.0,
        "match_rate_b": len(pairs) / len(B) if len(B) else 0.0,
        "sum_a": sum_a, "sum_b": sum_b, "gap": sum_a - sum_b,
        "matched_diff": md, "mismatch_diff": xd, "only_a_total": un_a, "only_b_total": un_b,
        "unexplained": unexplained,
        "bad_amount_a": int(A["std_amount"].isna().sum()),
        "bad_amount_b": int(B["std_amount"].isna().sum()),
        "bad_date_a": int(A["std_date"].isna().sum()) if A["std_date"].notna().any() else 0,
        "bad_date_b": int(B["std_date"].isna().sum()) if B["std_date"].notna().any() else 0,
        "by_type": pairs["match_type"].value_counts().to_dict(),
    }
    return {
        "summary": summary, "matched": matched, "mismatches": mismatches,
        "only_a": remA.reset_index(drop=True), "only_b": remB.reset_index(drop=True),
        "dupes_a": _dupes(A), "dupes_b": _dupes(B),
    }


# ------------------------------------------------------------------ export
def _sheet(name: str) -> str:
    return re.sub(r"[\[\]:*?/\\]", "", name)[:31]


def to_excel(res: dict, name_a: str, name_b: str) -> bytes:
    s = res["summary"]
    summary_df = pd.DataFrame([
        (f"Rows in {name_a}", s["rows_a"]), (f"Rows in {name_b}", s["rows_b"]),
        ("Matched pairs", s["matched"]), ("Amount mismatches", s["mismatched"]),
        (f"Only in {name_a}", s["only_a"]), (f"Only in {name_b}", s["only_b"]),
        (f"Total {name_a}", s["sum_a"]), (f"Total {name_b}", s["sum_b"]),
        ("Gap (A - B)", s["gap"]),
        ("  explained by: diffs inside tolerance", s["matched_diff"]),
        ("  explained by: amount mismatches", s["mismatch_diff"]),
        (f"  explained by: only in {name_a}", s["only_a_total"]),
        (f"  explained by: only in {name_b} (subtracted)", -s["only_b_total"]),
        ("  unexplained (should be 0)", s["unexplained"]),
    ], columns=["Item", "Value"])
    sheets = [
        ("Summary", summary_df), ("Matched", res["matched"]),
        ("Amount mismatches", res["mismatches"]),
        (f"Only in {name_a}", res["only_a"]), (f"Only in {name_b}", res["only_b"]),
        (f"Dupes {name_a}", res["dupes_a"]), (f"Dupes {name_b}", res["dupes_b"]),
    ]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        used = set()
        for nm, df in sheets:
            nm = _sheet(nm)
            while nm in used:
                nm = nm[:-1] + "_"
            used.add(nm)
            df.to_excel(xw, sheet_name=nm, index=False)
            ws = xw.sheets[nm]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                width = min(max(len(str(col)), 10) + 2, 40)
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
    return buf.getvalue()


# ------------------------------------------------------------------ sample
def make_sample(n: int = 150, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2026-10-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D")
    amt = np.round(rng.uniform(20, 5000, n), 2)
    ref = [f"INV-{1000 + i}" for i in range(n)]
    bank = pd.DataFrame({"Date": dates, "Reference": ref,
                         "Description": [f"Payment {r}" for r in ref], "Amount": amt})
    ledger = bank.copy()
    ledger.columns = ["Posting Date", "Doc No", "Narration", "Value"]
    idx = rng.permutation(n)
    drop, mism, tol_, noref, shift = idx[:8], idx[8:14], idx[14:20], idx[20:30], idx[30:36]
    ledger.loc[mism, "Value"] += np.round(rng.uniform(5, 50, len(mism)), 2)
    ledger.loc[tol_, "Value"] += 0.01
    ledger.loc[noref, "Doc No"] = ""
    ledger.loc[shift, "Posting Date"] += pd.Timedelta(days=2)
    ledger = ledger.drop(index=drop)
    extra = pd.DataFrame({"Posting Date": pd.Timestamp("2026-10-15"),
                          "Doc No": [f"GL-{900 + i}" for i in range(5)],
                          "Narration": "Manual journal",
                          "Value": [123.45, 678.9, 250.0, 99.99, 1500.0]})
    ledger = pd.concat([ledger, extra, ledger.iloc[[0]]], ignore_index=True)  # +5 extras, +1 duplicate
    return bank, ledger.sample(frac=1, random_state=1).reset_index(drop=True)

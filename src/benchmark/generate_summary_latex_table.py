#!/usr/bin/env python3
"""
Reads a CSV and writes out a LaTeX table with an adjustwidth environment and custom formatting:
- no 'n_seeds' column
- first column renamed to 'Method', extracting and formatting the active learning variant
- headers capitalized, acronyms fully uppercase with proper math mode for subscripts
- floats rounded to one decimal place, except CV rounded to two decimals
- LaTeX structure includes adjustwidth*, captionsetup, arraystretch, makebox, and custom tabularx spec

Usage:
    python3 generate_latex_table.py \
        --input summary.csv \
        --label tab:mytable \
        --caption "My table caption." \
        [--method-order Random,VARIQuery,DUO,Hybrid]
        > table.tex
"""
import argparse
import pandas as pd
import re
import sys

# Map acronyms to desired headers, with subscripts in math mode
ACRONYM_HEADERS = {
    'iqr': 'IQR',
    'std': 'STD',
    'sem': 'SEM',
    'ci_lower': r'CI$_{\ell}$',  # subscript l
    'ci_upper': r'CI$_{u}$',
    'cv': 'Cv'
}

# Map extracted method codes to display names
METHOD_NAME_MAP = {
    'duo_prio': 'DUO',
    'duo': 'DUO',
    'random': 'Random',
    'hybrid': 'Hybrid',
    'hybrid_prio_u_v6': 'Hybrid',
    'variquery': 'VARIQuery'
}

# Default row order for methods (can be overridden by --method-order)
DEFAULT_METHOD_ORDER = ['Random', 'VARIQuery', 'DUO', 'Hybrid']

def extract_method(prefix: str) -> str:
    """Extract and format method name from exp_name_prefix."""
    m = re.search(r'prefppo_([A-Za-z]+)', prefix)
    if m:
        code = m.group(1)
        return METHOD_NAME_MAP.get(code, code.title())
    return prefix


def format_value(val, col_key):
    """Format values: ints no decimals; floats one decimal normally; two decimals for CV."""
    if pd.isna(val):
        return ""
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if col_key == 'cv':
            return f"{val:.2f}"
        return f"{val:.1f}"
    return str(val)

def _apply_method_order(df: pd.DataFrame, order_list):
    """Sort rows by the 'Method' column using the provided order."""
    if 'Method' not in df.columns:
        return df
    # Append any methods not listed in order_list, preserving their first-appearance order
    remaining = [m for m in df['Method'].tolist() if m not in order_list]
    # Keep unique order for remaining
    seen = set()
    remaining_unique = [m for m in remaining if not (m in seen or seen.add(m))]
    full_order = order_list + remaining_unique

    cat = pd.Categorical(df['Method'], categories=full_order, ordered=True)
    df_sorted = df.copy()
    df_sorted['Method'] = cat
    df_sorted = df_sorted.sort_values('Method', kind='stable').reset_index(drop=True)
    # restore 'Method' as plain strings (not Categorical) for LaTeX output
    df_sorted['Method'] = df_sorted['Method'].astype(str)
    return df_sorted

def make_table(df: pd.DataFrame, label: str, caption: str, method_order=None) -> str:
    # Drop n_seeds
    df = df.drop(columns=[c for c in ['n_seeds', 'sem'] if c in df.columns])
    # Extract Method
    if 'exp_name_prefix' in df.columns:
        df.insert(0, 'Method', df['exp_name_prefix'].apply(extract_method))
        df = df.drop(columns=['exp_name_prefix'])

    if method_order is None or len(method_order) == 0:
        method_order = DEFAULT_METHOD_ORDER
    df = _apply_method_order(df, method_order)

    # Prepare
    col_keys = df.columns.tolist()
    # First column l, rest r
    col_spec = ['l'] + ['r'] * (len(col_keys)-1)
    col_spec_str = '@{}' + '|'.join(col_spec) + '@{}'
    # Headers
    headers = []
    for key in col_keys:
        kl = key.lower()
        if kl in ACRONYM_HEADERS:
            hdr = ACRONYM_HEADERS[kl]
        elif kl == 'method':
            hdr = 'Method'
        else:
            hdr = key.replace('_',' ').title()
        headers.append(f"\\textbf{{{hdr}}}")
    # Build lines
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \begin{adjustwidth*}{-3.25em}{-1.0em}")
    lines.append(r"    \centering")
    lines.append(r"    \setlength{\tabcolsep}{4pt}")
    lines.append(r"    \captionsetup{singlelinecheck=false, justification=raggedright}")
    lines.append(r"    \renewcommand{\arraystretch}{0.9}")
    lines.append(r"    \makebox[\linewidth][l]{%")
    lines.append(f"      \\begin{{tabularx}}{{\\linewidth}}{{{col_spec_str}}}")
    # header row
    hdr_line = ' & '.join(headers)
    lines.append(f"        {hdr_line} \\\\")
    # data rows
    for i, row in df.iterrows():
        is_last = (i == len(df) - 1)

        vals = [format_value(row[k], k.lower()) for k in col_keys]
        row_line = ' & '.join(vals)
        # hline = '\\\\ \\hline' if not is_last else ''
        lines.append(f"        {row_line} \\\\")
    lines.append(r"      \end{tabularx}%")
    lines.append(r"    }")
    lines.append(f"    \caption{{{caption}}}")
    lines.append(f"    \label{{{label}}}")
    lines.append(r"  \end{adjustwidth*}")
    lines.append(r"\end{table}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input',required=True)
    p.add_argument('-l','--label',required=True)
    p.add_argument('-c','--caption',required=True)
    p.add_argument('--method-order', default=",".join(DEFAULT_METHOD_ORDER),
                   help="Comma-separated order of methods (e.g., 'DUO,Hybrid,Random,VARIQuery').")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    order_list = [m.strip() for m in args.method_order.split(',') if m.strip()]
    tex = make_table(df, args.label, args.caption, method_order=order_list)
    sys.stdout.write(tex)


if __name__=='__main__':
    main()
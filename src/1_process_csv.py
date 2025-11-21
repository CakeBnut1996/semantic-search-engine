from sentence_transformers import SentenceTransformer
import pandas as pd
import os, hashlib, json, re
import numpy as np
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_datetime64_any_dtype, CategoricalDtype

# Notes Oct 20: Not sure how the user will query data.
# It is necessary to have descriptions in the db.

# === CONFIG ===
root = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
filename = 'billionton_23_macro_algae.csv' # 'pnnl-19944.pdf'
base = filename.split('.')[0]
CSV_PATH = os.path.join(root,'data science kdf','data',filename)

df = pd.read_csv(CSV_PATH, nrows=100)     # read first 100 rows only
print("Shape:", df.shape)

# === STATISTICAL SUMMARY GENERATOR ===
def generate_stats_summary(df, base):
    n_rows, n_cols = df.shape
    cols = df.columns.tolist()[:10]

    lines = [
        f"Dataset '{base}.csv' has {n_rows:,} rows and {n_cols} columns. Columns include: {', '.join(cols)}."
    ]

    try:
        desc = df.describe(include='all', datetime_is_numeric=True).transpose()
    except TypeError:
        desc = df.describe(include='all').transpose()

    stats = []
    for col in cols:
        if col not in desc.index:
            continue
        dtype = str(df[col].dtype)
        col_info = f"{col} ({dtype}): "

        if is_numeric_dtype(df[col]):
            mean = round(desc.loc[col, 'mean'], 3) if 'mean' in desc.columns else None
            median = df[col].median() if pd.notna(df[col].median()) else None
            col_info += f"mean={mean}, median={median}"

        elif is_object_dtype(df[col]) or isinstance(df[col].dtype, CategoricalDtype):
            top = desc.loc[col, 'top'] if 'top' in desc.columns else None
            freq = int(desc.loc[col, 'freq']) if 'freq' in desc.columns and pd.notna(desc.loc[col, 'freq']) else None
            if top:
                col_info += f"most frequent='{top}' ({freq} occurrences)"

        elif is_datetime64_any_dtype(df[col]):
            col_info += f"date range {df[col].min()} → {df[col].max()}"

        else:
            col_info += "no summary available"

        stats.append(col_info)

    lines.append(" | ".join(stats[:8]))
    return " ".join(lines)

# === EMBEDDING ===
def compute_embeddings(chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings


# Generate hybrid summary
stats_summary = generate_stats_summary(df, base)
text_summary = stats_summary

embedding = compute_embeddings([text_summary])[0]
print(f"Computed embeddings shape: {np.array(embedding).shape}")

# === STORE ===
out = [
    {
        "chunk": text_summary,
        "embedding": embedding.tolist()
    }
]
out_path = os.path.join(root,'data science kdf','embedding',base +'.json')
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"✅ Saved embeddings to {out_path}")

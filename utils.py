
import os
import pandas as pd
import chardet
from datetime import datetime

# ---------- 1. Detect File Encoding Safely ----------
def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        return result.get('encoding', 'utf-8')
    except Exception:
        return 'utf-8'

# ---------- 2. Read File with Detected Encoding ----------
def read_file_safely(file_path):
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        return df, f"✅ Successfully read file with encoding: {encoding}"
    except Exception as e:
        return None, f"❌ Failed to read file: {str(e)}"

# ---------- 3. Detect File Type Based on Column Keywords ----------
def detect_file_type(df):
    keyword_map = {
        "sales": ['product', 'revenue', 'sales', 'invoice'],
        "stock": ['stock', 'symbol', 'ticker', 'close', 'open', 'high', 'low'],
        "campaign": ['campaign', 'ads', 'impression', 'click', 'response', 'acceptedcmp'],
        "customer": ['customer', 'user', 'name', 'id', 'segment']
    }

    cols = " ".join(df.columns).lower()
    for file_type, keywords in keyword_map.items():
        if any(kw in cols for kw in keywords):
            return file_type
    return "generic"

# ---------- 4. Save Cleaned CSV with Type & Timestamp ----------
def save_cleaned_file(df, original_filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    file_type = detect_file_type(df)
    cleaned_filename = f"{file_type}_cleaned_{base_name}_{timestamp}.csv"

    output_dir = os.path.join("static", "cleaned")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cleaned_filename)

    try:
        df.to_csv(output_path, index=False)
        return cleaned_filename, f"✅ Cleaned data saved as: {cleaned_filename}"
    except Exception as e:
        return None, f"❌ Failed to save cleaned file: {str(e)}"

# ---------- 5. Save DataFrame as JSON (Optional Utility) ----------
def save_as_json(df, filename='data.json'):
    output_dir = os.path.join("static", "json")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        df.to_json(output_path, orient='records', lines=True)
        return f"✅ JSON saved to {output_path}"
    except Exception as e:
        return f"❌ Failed to export JSON: {str(e)}"

# ---------- 6. Generate Summary Dictionary for Reporting ----------
def summarize_dataframe(df):
    missing_percents = df.isnull().mean().round(3) * 100
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include='number').columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "date_columns": [col for col in df.columns if 'date' in col.lower()],
        "has_missing_values": df.isnull().any().any(),
        "missing_percent_per_column": missing_percents[missing_percents > 0].to_dict()
    }
    return summary

# ---------- 7. Generate Safe Filenames ----------
def safe_filename(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

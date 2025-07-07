import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import datetime

def detect_outliers(df):
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask[col] = (df[col] < lower) | (df[col] > upper)
    return outlier_mask

def process_data(df, missing_strategy='impute', outlier_strategy='replace_with_best'):
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'initial_shape': df.shape,
        'missing_values': df.isnull().sum().to_dict()
    }

    # Step 1: Convert datetime columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                continue

    # Step 2: Detect outliers
    outlier_mask = detect_outliers(df)
    report['outlier_counts'] = outlier_mask.sum().to_dict()

    # Step 3: Preview rows with issues
    affected_rows = df[outlier_mask.any(axis=1) | df.isnull().any(axis=1)]
    report['preview_rows'] = affected_rows.head(5).to_dict(orient='records')

    df_clean = df.copy()
    total_outliers_replaced = 0
    total_missing_filled = 0
    outlier_log = []
    missing_log = []

    # Step 4: Handle Outliers
    if outlier_strategy == 'drop_row':
        df_clean = df_clean[~outlier_mask.any(axis=1)]
        outlier_log.append("🚫 Dropped rows with outliers.")
    elif outlier_strategy in ['replace_with_mean', 'replace_with_best']:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            count = outlier_mask[col].sum()
            if count > 0:
                replacement = (
                    df_clean[col].median()
                    if outlier_strategy == 'replace_with_best'
                    else df_clean[col].mean()
                )
                df_clean.loc[outlier_mask[col], col] = replacement
                total_outliers_replaced += count
                outlier_log.append(
                    f"🔁 Replaced {count} outliers in '{col}' with {'median' if outlier_strategy == 'replace_with_best' else 'mean'} ({replacement:.2f})."
                )

    # Step 5: Handle Missing Values
    if missing_strategy == 'drop':
        before = df_clean.shape[0]
        df_clean = df_clean.dropna()
        dropped = before - df_clean.shape[0]
        total_missing_filled = dropped
        missing_log.append(f"🧹 Dropped {dropped} rows with missing values.")
    elif missing_strategy in ['impute', 'best']:
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns

        if len(num_cols) > 0:
            imputer_num = SimpleImputer(strategy='mean')
            missing_before = df_clean[num_cols].isnull().sum().sum()
            df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])
            if missing_before > 0:
                total_missing_filled += missing_before
                for col in num_cols:
                    if df[col].isnull().sum() > 0:
                        missing_log.append(f"🧠 Filled missing values in '{col}' with mean.")

        if len(cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            missing_before = df_clean[cat_cols].isnull().sum().sum()
            df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])
            if missing_before > 0:
                total_missing_filled += missing_before
                for col in cat_cols:
                    if df[col].isnull().sum() > 0:
                        missing_log.append(f"🗂️ Filled missing values in '{col}' with most frequent value.")

    # Step 6: Final Summary
    report['final_shape'] = df_clean.shape
    report['outliers_replaced'] = int(total_outliers_replaced)
    report['missing_filled'] = int(total_missing_filled)
    report['columns_with_issues'] = list(
        set(df.columns[df.isnull().any()]) |
        set(outlier_mask.columns[outlier_mask.any()])
    )
    report['logs'] = {
        "outlier_fixes": outlier_log or ["✅ No outliers found."],
        "missing_fixes": missing_log or ["✅ No missing values found."]
    }

    return report, df_clean
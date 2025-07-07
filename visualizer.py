import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Enables rendering without a display (server-safe)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set output directory for plots
PLOTS_DIR = 'static/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def safe_filename(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

def generate_visualizations(original_df, cleaned_df, max_columns=5):
    plots = []
    summary_log = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # 1. Missing Value Plot
    try:
        missing = original_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        total_missing = missing.sum()

        if not missing.empty:
            top_missing = missing[:max_columns]
            plt.figure(figsize=(12, 5))
            sns.barplot(x=top_missing.index, y=top_missing.values, palette='Oranges_r')
            plt.title("Top Missing Values Per Column")
            plt.ylabel("Missing Count")
            plt.xticks(rotation=45)
            plt.tight_layout()

            filename = f"missing_summary_{timestamp}.png"
            plt.savefig(os.path.join(PLOTS_DIR, filename))
            plt.close()
            plots.append(filename)

            summary_log.append(f"📊 Found {len(missing)} columns with missing values.")
            summary_log.append(f"🧮 Total missing cells: {total_missing}")
        else:
            summary_log.append("✅ No missing values found.")
    except Exception as e:
        summary_log.append(f"❌ Error in missing value plot: {str(e)}")

    # 2. Boxplots: Before vs. After Cleaning
    numeric_cols = original_df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        summary_log.append("⚠️ No numeric columns found for boxplots.")
    else:
        for col in numeric_cols[:max_columns]:
            try:
                if original_df[col].nunique(dropna=True) < 5:
                    summary_log.append(f"ℹ️ Skipped '{col}' (low variance or constant values).")
                    continue

                plt.figure(figsize=(10, 4))

                # Before cleaning
                plt.subplot(1, 2, 1)
                sns.boxplot(y=original_df[col].dropna(), color='salmon')
                plt.title(f"{col} - Before Cleaning")

                # After cleaning
                plt.subplot(1, 2, 2)
                sns.boxplot(y=cleaned_df[col].dropna(), color='lightgreen')
                plt.title(f"{col} - After Cleaning")

                plt.tight_layout()
                filename = f"{safe_filename(col)}_boxplot_{timestamp}.png"
                plt.savefig(os.path.join(PLOTS_DIR, filename))
                plt.close()
                plots.append(filename)

                summary_log.append(f"📦 Boxplot generated for '{col}'")
            except Exception as e:
                summary_log.append(f"❌ Boxplot error for '{col}': {str(e)}")

    # 3. Histograms (After Cleaning)
    for col in numeric_cols[:max_columns]:
        try:
            if cleaned_df[col].nunique(dropna=True) < 5:
                summary_log.append(f"ℹ️ Skipped histogram for '{col}' (low variance or constant values).")
                continue

            plt.figure(figsize=(6, 4))
            sns.histplot(cleaned_df[col].dropna(), kde=True, bins=20, color='skyblue')
            plt.title(f"{col} Distribution (After Cleaning)")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()

            filename = f"{safe_filename(col)}_dist_{timestamp}.png"
            plt.savefig(os.path.join(PLOTS_DIR, filename))
            plt.close()
            plots.append(filename)

            summary_log.append(f"📈 Histogram generated for '{col}'")
        except Exception as e:
            summary_log.append(f"❌ Histogram error for '{col}': {str(e)}")

    return {
        "plots": plots,
        "generated_at": timestamp,
        "log": summary_log or ["ℹ️ No plots generated."]
    }
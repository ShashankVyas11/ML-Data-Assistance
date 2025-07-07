from flask import Flask, render_template, request, redirect, send_from_directory, flash, url_for
import os
import pandas as pd

from data_processor import process_data
from model_trainer import train_and_predict
from visualizer import generate_visualizations
from utils import save_cleaned_file, read_file_safely

# ---------------------- Configuration ---------------------- #
UPLOAD_FOLDER = 'uploads'
CLEANED_FOLDER = 'static/cleaned'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------------- Routes ---------------------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("⚠️ No file selected. Please upload a CSV file.", "error")
            return redirect(url_for('index'))

        if not file.filename.endswith('.csv'):
            flash("❌ Invalid file type. Only .csv files are supported.", "error")
            return redirect(url_for('index'))

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"[INFO] ✅ Uploaded file saved at: {filepath}")

        # Safe read
        df, read_msg = read_file_safely(filepath)
        if df is None:
            flash(read_msg, "error")
            return redirect(url_for('index'))

        df.columns = df.columns.str.strip()
        print(f"[INFO] {read_msg}")

        # Strategies
        missing_strategy = request.form.get('missing_strategy', 'impute')
        outlier_strategy = request.form.get('outlier_strategy', 'replace_with_mean')

        # Clean data
        report, cleaned_df = process_data(df, missing_strategy, outlier_strategy)

        # Generate visualizations
        viz_result = generate_visualizations(df, cleaned_df)
        plots = viz_result.get('plots', [])
        print(f"[INFO] 🖼️ {len(plots)} plots generated.")

        # Save cleaned data
        cleaned_filename, save_msg = save_cleaned_file(cleaned_df, filename)
        print(f"[INFO] {save_msg}")

        return render_template('analysis.html',
                               report=report,
                               plots=plots,
                               filename=cleaned_filename,
                               upload_time=report['timestamp'])

    except Exception as e:
        print(f"❌ [ERROR] Analysis failed: {e}")
        return render_template('error.html', error=f"Data analysis error: {str(e)}")

@app.route('/predict/<filename>')
def predict(filename):
    try:
        filepath = os.path.join(CLEANED_FOLDER, filename)
        if not os.path.exists(filepath):
            flash("⚠️ File not found for prediction.", "error")
            return redirect(url_for('index'))

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        results = train_and_predict(df)

        return render_template('prediction.html',
                               results=results,
                               filename=filename)

    except Exception as e:
        print(f"❌ [ERROR] Prediction failed: {e}")
        return render_template('error.html', error=f"Prediction error: {str(e)}")

@app.route('/download/<filename>')
def download(filename):
    try:
        return send_from_directory(CLEANED_FOLDER, filename, as_attachment=True)
    except Exception as e:
        print(f"❌ [ERROR] Download failed: {e}")
        return render_template('error.html', error=f"Download error: {str(e)}")

@app.route('/reset')
def reset():
    return redirect(url_for('index'))

# ---------------------- Run Server ---------------------- #
if __name__ == '__main__':
    app.run(debug=True)

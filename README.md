**ML Data Assistance**

1. ML Data Assistance is an intelligent web-based data assistant built using Flask and machine learning. It automates the process of **data cleaning, analysis, and prediction** — saving time for data analysts, scientists, and business users.

##  Key Features

### 🧹 1. Smart Data Cleaning
- Automatically detects and handles:
  - **Missing values** using techniques like mean, median, or KNN-based imputation.
  - **Outliers** using statistical or clustering-based methods.
- Supports options to:
  - Remove or replace missing/outlier values.
  - Visualize the data **before and after cleaning**.
- Generates a **cleaning summary report**:
  - Number of missing values detected & handled.
  - Number of outliers detected & handled.
  - Downloadable cleaned file in `.csv` format.

### 📊 2. Data Visualization & Reporting
- Dynamic charts (Matplotlib & Seaborn) to show:
  - Missing value heatmaps
  - Distribution plots
  - Before vs After comparison of cleaned data
- Clean, professional interface to explore visual insights.

### 📈 3. Auto ML Prediction for Sales, Stock & Customer Data
- Recognizes structured data types like:
  - **Sales data** → Predicts next 6 months' sales using ML models (Random Forest, XGBoost, Prophet)
  - **Stock/Inventory data** → Detects trends, anomalies, and forecasts using time series models
  - **Customer data** → Recommends top buyers, high-value regions, and product interest

- Automatically predicts and displays:
  - 🔝 Top 5 selling products
  - 🌍 Top 5 locations by sales
  - 📉 Predicted sales trends
  - 👥 Best customers

## 🛠️ Tech Stack

| Layer            | Technologies                                                                 |
|------------------|------------------------------------------------------------------------------|
| 🧩 Backend        | Python, Flask                                                               |
| 📊 Data Handling  | Pandas, NumPy, scikit-learn, XGBoost, Prophet                                |
| 📈 Visualization  | Matplotlib, Seaborn, Plotly                                                 |
| 🔧 Utilities      | chardet, openpyxl, joblib                                                   |

## 💼 Skills Demonstrated

- ✔️ Data Cleaning (Missing value imputation, Outlier detection/removal)
- ✔️ Exploratory Data Analysis (EDA)
- ✔️ ML Model Training & Forecasting (Regression, Time Series)
- ✔️ Data Visualization
- ✔️ Flask Web App Development
- ✔️ File Handling & Report Generation
- ✔️ Deployment-Ready Project

## 📂 Project Structure
ml-data-assistance/
│
├── app.py # Main Flask app
├── model_trainer.py # ML model training and prediction logic
├── data_processor.py # Cleaning, EDA, feature handling
├── visualizer.py # Plot generation and comparison visuals
├── utils.py # Helper functions (e.g., safe file saving)
├── templates/ # HTML templates (Jinja2)
├── static/ # CSS, JS, and image assets
├── requirements.txt # Python dependencies
└── README.md # You're here!

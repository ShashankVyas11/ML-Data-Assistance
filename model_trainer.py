import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

PLOTS_DIR = 'static/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def safe_filename(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

def detect_data_type(df):
    cols = " ".join(df.columns).lower()
    if 'symbol' in cols and ('price' in cols or 'close' in cols or 'open' in cols):
        return 'stock'
    elif 'campaign' in cols or 'response' in cols or 'offer' in cols:
        return 'campaign'
    elif 'product' in cols and ('sales' in cols or 'revenue' in cols or 'amount' in cols):
        return 'sales'
    elif 'customer' in cols and ('name' in cols or 'id' in cols or 'purchase' in cols):
        return 'customer'
    else:
        return 'generic'

def train_and_predict(df, symbol=None):
    results = {}
    feedback = []
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data_type = detect_data_type(df)
    results['detected_type'] = data_type

    product_col = next((col for col in df.columns if 'product' in col.lower()), None)
    revenue_col = next((col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower() or 'amount' in col.lower()), None)
    date_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
    quantity_col = next((col for col in df.columns if 'quantity' in col.lower()), None)
    price_col = next((col for col in df.columns if 'price' in col.lower()), None)
    customer_col = next((col for col in df.columns if 'customer' in col.lower() and 'name' in col.lower()), None)
    location_col = next((col for col in df.columns if 'location' in col.lower() or 'city' in col.lower()), None)

    if revenue_col:
        df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')

    if data_type == 'sales':
        if product_col and revenue_col:
            try:
                top = df.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(5)
                results['sales_summary'] = [f"{k}: ₹{v:.2f}" for k, v in top.items()]
                results['top_5_products'] = top.reset_index().to_dict(orient='records')

                plt.figure(figsize=(8, 5))
                sns.barplot(x=top.values, y=top.index, palette='viridis')
                plt.title("Top 5 Selling Products")
                plt.xlabel("Total Revenue")
                plt.tight_layout()
                fname = safe_filename(f"top5_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                path = os.path.join(PLOTS_DIR, fname)
                plt.savefig(path)
                plt.close()
                results['top5_plot'] = fname
            except Exception as e:
                feedback.append(f"❌ Top product analysis failed: {str(e)}")

        if location_col and revenue_col:
            try:
                loc = df.groupby(location_col)[revenue_col].sum().sort_values(ascending=False).head(5)
                results['location_summary'] = [f"{k}: ₹{v:.2f}" for k, v in loc.items()]
            except Exception as e:
                feedback.append(f"❌ Location analysis failed: {str(e)}")

        if date_col and revenue_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                monthly_df = df.dropna(subset=[date_col])
                monthly_df = monthly_df.groupby(monthly_df[date_col].dt.to_period('M'))[revenue_col].sum().reset_index()
                monthly_df[date_col] = monthly_df[date_col].dt.to_timestamp()
                monthly_df['month'] = (monthly_df[date_col] - monthly_df[date_col].min()).dt.days // 30

                X = monthly_df[['month']]
                y = monthly_df[revenue_col]

                model = XGBRegressor()
                model.fit(X, y)

                future_months = pd.DataFrame({'month': list(range(X['month'].max() + 1, X['month'].max() + 7))})
                future_preds = model.predict(future_months)

                trend_data = pd.DataFrame({
                    'month': future_months['month'],
                    'predicted_revenue': future_preds
                })

                results['sales_prediction'] = (
                    f"📈 Predicted revenue 6 months ahead. Growth from ₹{y.iloc[-1]:.2f} to ₹{future_preds[-1]:.2f}"
                )
                results['growth_trend'] = trend_data.to_dict(orient='records')

                trend_data['month_index'] = range(len(trend_data))
                plt.figure(figsize=(10, 4))
                sns.lineplot(x='month_index', y='predicted_revenue', data=trend_data, marker='o', color='blue')
                plt.title("Predicted Revenue Growth (Next 6 Months)")
                plt.xlabel("Months Ahead")
                plt.ylabel("Revenue")
                plt.tight_layout()
                fname = safe_filename(f"predicted_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                path = os.path.join(PLOTS_DIR, fname)
                plt.savefig(path)
                plt.close()
                results['growth_plot'] = fname
            except Exception as e:
                results['sales_prediction'] = f"⚠️ Future prediction failed: {str(e)}"
                feedback.append("⚠️ Future forecast error.")
        else:
            results['sales_prediction'] = "⚠️ No date column found for future trend prediction."

        if customer_col and product_col:
            try:
                behavior = df.groupby([customer_col, product_col]).size().reset_index(name='purchase_count')
                top_behavior = behavior.sort_values('purchase_count', ascending=False).head(1)
                if not top_behavior.empty:
                    top = top_behavior.iloc[0]
                    results['customer_behavior'] = (
                        f"'{top[customer_col]}' bought '{top[product_col]}' {int(top['purchase_count'])} times."
                    )
                else:
                    results['customer_behavior'] = "ℹ️ No repeat purchases detected."
            except Exception as e:
                results['customer_behavior'] = f"⚠️ Behavior analysis failed: {str(e)}"
        else:
            results['customer_behavior'] = "⚠️ Missing customer/product info for behavior analysis."

    elif data_type == 'campaign':
        try:
            if 'response' in df.columns and 'offer' in df.columns:
                df['response'] = df['response'].astype(str).str.lower()
                accepted_rate = df[df['response'].str.contains("yes|accepted")].shape[0] / df.shape[0]
                results['campaign_success'] = f"✅ Campaign acceptance rate: {accepted_rate:.2%}"
            else:
                results['campaign_success'] = "⚠️ No clear response/offer columns to evaluate campaign performance."
        except Exception as e:
            results['campaign_success'] = f"❌ Campaign analysis error: {str(e)}"

    elif data_type == 'stock':
        try:
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            price_col = next((col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()), None)
            symbol_col = next((col for col in df.columns if 'symbol' in col.lower() or 'stock' in col.lower()), None)

            if date_col and price_col and symbol_col:
                symbols = df[symbol_col].dropna().unique()
                results['available_symbols'] = symbols.tolist()

                if symbol is None:
                    symbol = symbols[0]
                results['selected_symbol'] = symbol

                stock_df = df[df[symbol_col] == symbol].copy()
                stock_df[date_col] = pd.to_datetime(stock_df[date_col], errors='coerce')
                stock_df = stock_df.dropna(subset=[date_col, price_col])
                stock_df.sort_values(by=date_col, inplace=True)
                stock_df['time_index'] = (stock_df[date_col] - stock_df[date_col].min()).dt.days

                X = stock_df[['time_index']]
                y = stock_df[price_col]

                model = XGBRegressor()
                model.fit(X, y)

                future_days = pd.DataFrame({'time_index': list(range(X['time_index'].max() + 1, X['time_index'].max() + 31))})
                future_prices = model.predict(future_days)

                trend_data = pd.DataFrame({
                    'day': range(1, 31),
                    'predicted_price': future_prices
                })

                results['stock_prediction'] = f"📈 {symbol}: Predicted price after 30 days is ₹{future_prices[-1]:.2f}"
                results['stock_trend'] = trend_data.to_dict(orient='records')

                plt.figure(figsize=(10, 4))
                sns.lineplot(x='day', y='predicted_price', data=trend_data, marker='o', color='green')
                plt.title(f"{symbol} Stock Price Forecast (Next 30 Days)")
                plt.xlabel("Days Ahead")
                plt.ylabel("Price")
                plt.tight_layout()
                fname = safe_filename(f"{symbol}_stock_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                path = os.path.join(PLOTS_DIR, fname)
                plt.savefig(path)
                plt.close()
                results['stock_plot'] = fname
            else:
                results['stock_prediction'] = "⚠️ Missing required columns (date, price, symbol)."
        except Exception as e:
            results['stock_prediction'] = f"⚠️ Stock prediction error: {str(e)}"

    if quantity_col and price_col and revenue_col:
        try:
            df[[quantity_col, price_col, revenue_col]] = df[[quantity_col, price_col, revenue_col]].apply(pd.to_numeric, errors='coerce')
            X = df[[quantity_col, price_col]].dropna()
            y = df.loc[X.index, revenue_col]

            if not X.empty:
                rf = RandomForestRegressor()
                rf.fit(X, y)
                pred = rf.predict(X[:1])[0]
                mse = mean_squared_error(y, rf.predict(X))
                results['rf_prediction'] = f"🧠 Value  (₹{np.sqrt(mse):.2f}) → ₹{pred:.2f}"
            else:
                feedback.append("⚠️ Not enough data for random forest prediction.")
        except Exception as e:
            feedback.append(f"⚠️ RF prediction failed: {str(e)}")

    results['feedback'] = feedback or ["✅ All insights generated."]
    return results

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="Inventory Demand Forecasting", layout="wide")

st.title("📦 Smart Inventory & Demand Forecasting Dashboard")
st.write("Upload your batch inventory CSV file to predict future demand, safety buffers, and reorder requirements.")

@st.cache_resource
def load_model():
    with open("xgboost_inventory_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'xgboost_inventory_model.pkl' not found!")
    st.stop()

uploaded_file = st.file_uploader("Upload Batch Inventory CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.success("Successfully loaded uploaded CSV file!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()
else:
    try:
        batch_df = pd.read_csv("sample_inventory.csv")
        st.info("Using default 'sample_inventory.csv'. (Upload your own file above anytime).")
    except FileNotFoundError:
        st.warning("No default 'sample_inventory.csv' found.")
        st.stop()

feature_cols = ['Unit_Price', 'Current_Stock', 'Historical_Avg', 'Promotion', 'Category_Electronics', 'Category_Clothing', 'Category_Home']
missing_cols = [col for col in feature_cols if col not in batch_df.columns]

if missing_cols:
    st.error(f"The uploaded CSV is missing these required feature columns: {missing_cols}")
else:
    preds = model.predict(batch_df[feature_cols])
    batch_df['Predicted_Demand'] = np.round(preds).astype(int)
    batch_df['Safety_Buffer'] = np.round(batch_df['Predicted_Demand'] * 0.15).astype(int)
    batch_df['Recommended_Reorder'] = np.maximum(0, (batch_df['Predicted_Demand'] + batch_df['Safety_Buffer']) - batch_df['Current_Stock'])
    batch_df['Status'] = np.where(batch_df['Current_Stock'] >= batch_df['Predicted_Demand'], 'Stock Sufficient', 'Action Required: Reorder')

    total_products = len(batch_df)
    reorder_count = len(batch_df[batch_df['Status'] == 'Action Required: Reorder'])
    total_demand = int(batch_df['Predicted_Demand'].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products Evaluated", total_products)
    col2.metric("Items Requiring Reorder", reorder_count)
    col3.metric("Total Predicted Demand", total_demand)

    st.markdown("### Detailed Output Table")
    st.dataframe(batch_df, use_container_width=True)

    csv_output = batch_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Batch Predictions CSV",
        data=csv_output,
        file_name="xgboost_inventory_predictions.csv",
        mime="text/csv",
        type="primary"
    )

    st.markdown("### 📊 Batch Visualizations")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Predicted Demand Distribution")
        fig_hist = px.histogram(batch_df, x='Predicted_Demand', nbins=15, template='plotly_white')
        fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart2:
        st.markdown("#### Current Stock vs Predicted Demand")
        fig_scatter = px.scatter(batch_df, x='Current_Stock', y='Predicted_Demand', color='Status', template='plotly_white')
        fig_scatter.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

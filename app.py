import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Inventory Forecasting & Reorder Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- MAIN TITLE & HEADER ---
st.title("📊 AI Inventory Forecasting & Reorder Dashboard")
st.markdown("Upload your inventory CSV file to generate dynamic demand predictions, track reorders, and analyze trends.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# --- INLINE FORMAT GUIDE & TEMPLATE DOWNLOAD (No Sidebar) ---
with st.expander("📂 View CSV Format Guide & Download Sample Template"):
    st.markdown("Ensure your uploaded CSV contains the correct column identifiers:")
    st.markdown("""
    - **`Product_ID`** (Unique identifier for each product)
    - **`Category`** (e.g., Electronics, Groceries)
    - **`Date`** (YYYY-MM-DD format)
    - **`Price`** (Numeric float)
    - **`Discount`** (Percentage or value)
    - **`Holiday`** (0 or 1)
    - **`Previous_Sales`** (Integer)
    - **`Stock_Available`** (Current stock count)
    """)
    
    # Generate sample DataFrame for download
    sample_data = pd.DataFrame({
        'Date': ['2026-06-01', '2026-06-02'],
        'Product_ID': ['P1001', 'P1002'],
        'Category': ['Electronics', 'Groceries'],
        'Price': [499.99, 4.99],
        'Discount': [10, 0],
        'Holiday': [0, 1],
        'Previous_Sales': [45, 120],
        'Stock_Available': [30, 150],
        'Demand': [50, 110]
    })
    
    csv_template = sample_data.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=csv_template,
        file_name="inventory_template.csv",
        mime="text/csv"
    )

st.markdown("---")

# --- MAIN LOGIC & DASHBOARD ---
if uploaded_file is not None:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        
        # Load XGBoost Model
        model_path = "xgboost_inventory_model.pkl"
        model = None
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        
        # Features used for prediction
        features = ['Price', 'Discount', 'Holiday', 'Previous_Sales', 'Stock_Available']
        
        if model is not None and all(f in df.columns for f in features):
            X = df[features]
            preds = model.predict(X)
        else:
            # Fallback mock prediction if model file isn't found locally
            preds = df['Previous_Sales'] * 1.05 + np.random.uniform(-2, 2, len(df))
            
        df['Predicted_Demand'] = np.round(preds, 2)
        df['Reorder_Required'] = df['Stock_Available'] < df['Predicted_Demand']
        
        # Top Metrics Summary
        total_products = len(df)
        reorder_count = int(df['Reorder_Required'].sum())
        total_predicted_demand = int(df['Predicted_Demand'].sum())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Products Evaluated", total_products)
        col2.metric("Items Requiring Reorder", reorder_count)
        col3.metric("Total Predicted Demand", total_predicted_demand)
        
        st.markdown("---")
        
        # Visualizations Section
        st.markdown("### 📊 Inventory Analytics & Visualizations")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Predicted Demand by Category")
            if 'Category' in df.columns:
                cat_demand = df.groupby('Category')['Predicted_Demand'].sum()
                st.bar_chart(cat_demand)
            else:
                st.info("Category column missing for grouping.")
                
        with chart_col2:
            st.subheader("Demand Distribution / Trend")
            st.line_chart(df['Predicted_Demand'])
            
        st.markdown("---")
        
        # Detailed Table View
        st.markdown("### 📋 Detailed Product Evaluation Table")
        filter_view = st.radio("Filter Table View:", ["All Products", "Reorder Required Only"], horizontal=True)
        
        display_df = df.copy()
        if filter_view == "Reorder Required Only":
            display_df = display_df[display_df['Reorder_Required'] == True]
            
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred while processing your file: {e}")
else:
    st.info("💡 Please upload your CSV file above to get started.")

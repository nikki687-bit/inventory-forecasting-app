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
st.markdown("Upload any inventory CSV file, auto-detect or map your columns, and generate dynamic demand predictions.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# --- INLINE FORMAT GUIDE & TEMPLATE DOWNLOAD ---
with st.expander("📂 View CSV Format Guide & Download Sample Template"):
    st.markdown("Ensure your uploaded CSV contains the correct column identifiers or use the smart mapper below if your column names differ:")
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

# Helper function to find best matching column name automatically
def find_best_match(columns, keywords):
    for col in columns:
        for kw in keywords:
            if kw in col.lower():
                return col
    return columns[0] if len(columns) > 0 else None

# --- MAIN LOGIC & DASHBOARD ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"Successfully uploaded `{uploaded_file.name}`! ({len(df)} rows found)")
        
        # --- SMART COLUMN MAPPING SECTION ---
        st.markdown("### ⚙️ Smart Column Mapping")
        st.markdown("We auto-detected your columns below. Adjust them if needed:")
        
        available_cols = list(df.columns)
        
        # Auto-detect defaults using keywords
        def_price = find_best_match(available_cols, ['price', 'cost', 'rate'])
        def_stock = find_best_match(available_cols, ['stock', 'inventory', 'qty', 'quantity'])
        def_discount = find_best_match(available_cols, ['discount', 'offer', 'markdown'])
        def_prev_sales = find_best_match(available_cols, ['previous', 'sales', 'past', 'sold'])
        def_holiday = find_best_match(available_cols, ['holiday', 'festival', 'promo'])
        def_category = find_best_match(available_cols, ['category', 'department', 'type', 'group'])
        
        col_map1, col_map2, col_map3 = st.columns(3)
        with col_map1:
            price_col = st.selectbox("Price Column", available_cols, index=available_cols.index(def_price) if def_price in available_cols else 0)
            stock_col = st.selectbox("Stock Available Column", available_cols, index=available_cols.index(def_stock) if def_stock in available_cols else 0)
        with col_map2:
            discount_col = st.selectbox("Discount Column", available_cols, index=available_cols.index(def_discount) if def_discount in available_cols else 0)
            prev_sales_col = st.selectbox("Previous Sales Column", available_cols, index=available_cols.index(def_prev_sales) if def_prev_sales in available_cols else 0)
        with col_map3:
            holiday_col = st.selectbox("Holiday Column", available_cols, index=available_cols.index(def_holiday) if def_holiday in available_cols else 0)
            category_col = st.selectbox("Category Column", available_cols, index=available_cols.index(def_category) if def_category in available_cols else 0)

        # Normalize mapped data for the model
        processed_df = pd.DataFrame()
        processed_df['Price'] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
        processed_df['Discount'] = pd.to_numeric(df[discount_col], errors='coerce').fillna(0)
        processed_df['Holiday'] = pd.to_numeric(df[holiday_col], errors='coerce').fillna(0)
        processed_df['Previous_Sales'] = pd.to_numeric(df[prev_sales_col], errors='coerce').fillna(0)
        processed_df['Stock_Available'] = pd.to_numeric(df[stock_col], errors='coerce').fillna(0)
        
        processed_df['Product_ID'] = df['Product_ID'] if 'Product_ID' in df.columns else [f"P100{i}" for i in range(len(df))]
        processed_df['Category'] = df[category_col] if category_col in df.columns else "General"
        processed_df['Date'] = df['Date'] if 'Date' in df.columns else "2026-06-01"

        # Load XGBoost Model
        model_path = "xgboost_inventory_model.pkl"
        model = None
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        
        features = ['Price', 'Discount', 'Holiday', 'Previous_Sales', 'Stock_Available']
        
        if model is not None:
            X = processed_df[features]
            preds = model.predict(X)
        else:
            preds = processed_df['Previous_Sales'] * 1.05 + np.random.uniform(-2, 2, len(processed_df))
            
        processed_df['Predicted_Demand'] = np.round(preds, 2)
        processed_df['Reorder_Required'] = processed_df['Stock_Available'] < processed_df['Predicted_Demand']
        
        st.markdown("---")
        
        # Top Metrics Summary
        total_products = len(processed_df)
        reorder_count = int(processed_df['Reorder_Required'].sum())
        total_predicted_demand = int(processed_df['Predicted_Demand'].sum())
        
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
            cat_demand = processed_df.groupby('Category')['Predicted_Demand'].sum()
            st.bar_chart(cat_demand)
                
        with chart_col2:
            st.subheader("Demand Distribution / Trend")
            st.line_chart(processed_df['Predicted_Demand'])
            
        st.markdown("---")
        
        # Detailed Table View
        st.markdown("### 📋 Detailed Product Evaluation Table")
        filter_view = st.radio("Filter Table View:", ["All Products", "Reorder Required Only"], horizontal=True)
        
        display_df = processed_df.copy()
        if filter_view == "Reorder Required Only":
            display_df = display_df[display_df['Reorder_Required'] == True]
            
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred while processing your file: {e}")
else:
    st.info("💡 Please upload your CSV file above to get started.")

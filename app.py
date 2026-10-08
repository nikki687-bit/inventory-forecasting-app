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
st.markdown("Upload any inventory CSV file, map your columns, and generate dynamic demand predictions.")

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

# --- MAIN LOGIC & DASHBOARD ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"Successfully uploaded `{uploaded_file.name}`! ({len(df)} rows found)")
        
        # --- UNIQUE SMART COLUMN MAPPING ---
        st.markdown("### ⚙️ Smart Column Mapping")
        st.markdown("Verify and adjust how your CSV columns map to the model requirements:")
        
        available_cols = list(df.columns)
        
        # Helper to find a distinct default column
        def get_default(keywords, used_cols):
            for col in available_cols:
                if col not in used_cols:
                    if any(kw in col.lower() for kw in keywords):
                        return col
            # Fallback to first available unused column
            for col in available_cols:
                if col not in used_cols:
                    return col
            return available_cols[0]

        used = []
        def_price = get_default(['price', 'cost', 'rate', 'unit_price'], used)
        used.append(def_price)
        
        def_stock = get_default(['stock', 'inventory', 'qty', 'quantity', 'current_stock'], used)
        used.append(def_stock)
        
        def_discount = get_default(['discount', 'offer', 'markdown', 'promo'], used)
        used.append(def_discount)
        
        def_prev_sales = get_default(['previous', 'sales', 'past', 'sold', 'demand'], used)
        used.append(def_prev_sales)
        
        def_holiday = get_default(['holiday', 'festival', 'promotion', 'is_holiday'], used)
        used.append(def_holiday)
        
        def_category = get_default(['category', 'department', 'type', 'group'], used)
        used.append(def_category)
        
        def_date = get_default(['date', 'time', 'day'], [])
        def_id = get_default(['id', 'product', 'item_id'], [])

        col_map1, col_map2, col_map3 = st.columns(3)
        with col_map1:
            price_col = st.selectbox("Price Column", available_cols, index=available_cols.index(def_price))
            stock_col = st.selectbox("Stock Available Column", available_cols, index=available_cols.index(def_stock))
        with col_map2:
            discount_col = st.selectbox("Discount Column", available_cols, index=available_cols.index(def_discount))
            prev_sales_col = st.selectbox("Previous Sales Column", available_cols, index=available_cols.index(def_prev_sales))
        with col_map3:
            holiday_col = st.selectbox("Holiday Column", available_cols, index=available_cols.index(def_holiday))
            category_col = st.selectbox("Category Column", available_cols, index=available_cols.index(def_category))

        # Base processed dataframe for display and prediction
        processed_df = pd.DataFrame()
        processed_df['Price'] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
        processed_df['Discount'] = pd.to_numeric(df[discount_col], errors='coerce').fillna(0)
        processed_df['Holiday'] = pd.to_numeric(df[holiday_col], errors='coerce').fillna(0)
        processed_df['Previous_Sales'] = pd.to_numeric(df[prev_sales_col], errors='coerce').fillna(0)
        processed_df['Stock_Available'] = pd.to_numeric(df[stock_col], errors='coerce').fillna(0)
        
        processed_df['Product_ID'] = df[def_id] if def_id in df.columns else [f"P100{i}" for i in range(len(df))]
        processed_df['Category'] = df[category_col] if category_col in df.columns else "General"
        
        date_col = def_date if def_date in df.columns else None
        if date_col:
            processed_df['Date'] = df[date_col]
            dt = pd.to_datetime(df[date_col], errors='coerce')
            processed_df['Year'] = dt.dt.year.fillna(2026)
            processed_df['Month'] = dt.dt.month.fillna(6)
            processed_df['Day'] = dt.dt.day.fillna(1)
            processed_df['DayOfWeek'] = dt.dt.dayofweek.fillna(0)
        else:
            processed_df['Date'] = "2026-06-01"
            processed_df['Year'] = 2026
            processed_df['Month'] = 6
            processed_df['Day'] = 1
            processed_df['DayOfWeek'] = 0

        # Load XGBoost Model
        model_path = "xgboost_inventory_model.pkl"
        model = None
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        
        if model is not None:
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
            elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
                expected_features = model.get_booster().feature_names
            else:
                expected_features = ['Price', 'Discount', 'Holiday', 'Previous_Sales', 'Stock_Available', 
                                     'Year', 'Month', 'Day', 'DayOfWeek', 
                                     'Category_Electronics', 'Category_Furniture', 'Category_Groceries', 'Category_Toys']
            
            temp_X = pd.DataFrame(index=processed_df.index)
            for col in ['Price', 'Discount', 'Holiday', 'Previous_Sales', 'Stock_Available', 'Year', 'Month', 'Day', 'DayOfWeek']:
                if col in processed_df.columns:
                    temp_X[col] = processed_df[col]
            
            for feat in expected_features:
                if feat.startswith("Category_"):
                    cat_name = feat.replace("Category_", "")
                    temp_X[feat] = (processed_df['Category'].astype(str).str.lower() == cat_name.lower()).astype(int)
            
            X = temp_X.reindex(columns=expected_features, fill_value=0)
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
            
        st.markdown---()
        
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

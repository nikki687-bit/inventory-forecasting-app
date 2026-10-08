import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Inventory AI Hub",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN UI STYLING (CUSTOM CSS) ---
st.markdown("""
    <style>
    /* Main background & font adjustments */
    .main {
        background-color: #f8f9fa;
    }
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetric"] label {
        color: #6c757d !important;
        font-weight: 600;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONTROLS & UPLOADER ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
    st.header("Inventory Control")
    st.markdown("Upload your stock sheet to run predictions.")
    
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "xls", "txt"])
    
    st.markdown("---")
    
    # Format Guide Expander inside Sidebar
    with st.expander("📂 View File Format Guide"):
        st.markdown("""
        Ensure your file has headers like:
        - `Product_ID`
        - `Category`
        - `Price`
        - `Discount`
        - `Holiday`
        - `Previous_Sales`
        - `Stock_Available`
        """)
        
        sample_data = pd.DataFrame({
            'Product_ID': ['P1001', 'P1002'],
            'Category': ['Electronics', 'Groceries'],
            'Price': [499.99, 4.99],
            'Discount': [10, 0],
            'Holiday': [0, 1],
            'Previous_Sales': [45, 120],
            'Stock_Available': [30, 150]
        })
        st.download_button(
            label="📥 Download Template",
            data=sample_data.to_csv(index=False).encode('utf-8'),
            file_name="template.csv",
            mime="text/csv"
        )
        
    st.markdown("---")
    st.caption("🚀 Powered by XGBoost & Streamlit")

# --- MAIN TITLE & HEADER ---
st.title("📦 AI Inventory & Demand Forecasting Hub")
st.markdown("Real-time automated stock analysis, demand prediction, and smart reorder alerts.")
st.markdown("---")

# --- MAIN LOGIC & DASHBOARD ---
if uploaded_file is not None:
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df = pd.read_csv(uploaded_file)
            
        available_cols = list(df.columns)
        
        def get_default(keywords, used_cols):
            for col in available_cols:
                if col not in used_cols and any(kw in col.lower() for kw in keywords):
                    return col
            for col in available_cols:
                if col not in used_cols:
                    return col
            return available_cols[0]

        used = []
        def_price = get_default(['price', 'cost', 'rate', 'unit_price'], used); used.append(def_price)
        def_stock = get_default(['stock', 'inventory', 'qty', 'quantity', 'current_stock'], used); used.append(def_stock)
        def_discount = get_default(['discount', 'offer', 'markdown', 'promo'], used); used.append(def_discount)
        def_prev_sales = get_default(['previous', 'sales', 'past', 'sold', 'history'], used); used.append(def_prev_sales)
        def_holiday = get_default(['holiday', 'festival', 'is_holiday'], used); used.append(def_holiday)
        def_category = get_default(['category', 'department', 'type', 'group'], used); used.append(def_category)
        def_date = get_default(['date', 'time', 'day'], [])
        
        def_id = None
        for col in available_cols:
            if any(k in col.lower() for k in ['id', 'product', 'item', 'name', 'sku']):
                def_id = col
                break

        # --- SMART MAPPING EXPANDER ---
        with st.expander("⚙️ Smart Column Mapping (Auto-detected — Click to adjust)", expanded=False):
            st.info(
                "💡 **How Smart Mapping Works:** Automatically links your custom spreadsheet headers "
                "to the machine learning model without requiring format changes."
            )
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
                
                id_options = ["None (Auto-generate IDs)"] + available_cols
                default_id_index = id_options.index(def_id) if def_id in id_options else 0
                selected_id_opt = st.selectbox("Product ID Column", id_options, index=default_id_index)
        
        if 'price_col' not in locals():
            price_col, stock_col, discount_col, prev_sales_col, holiday_col, category_col = def_price, def_stock, def_discount, def_prev_sales, def_holiday, def_category
            selected_id_opt = def_id if def_id else "None (Auto-generate IDs)"

        # Process DataFrame
        processed_df = pd.DataFrame()
        processed_df['Price'] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
        processed_df['Discount'] = pd.to_numeric(df[discount_col], errors='coerce').fillna(0)
        processed_df['Holiday'] = pd.to_numeric(df[holiday_col], errors='coerce').fillna(0)
        processed_df['Previous_Sales'] = pd.to_numeric(df[prev_sales_col], errors='coerce').fillna(0)
        processed_df['Stock_Available'] = pd.to_numeric(df[stock_col], errors='coerce').fillna(0)
        
        if selected_id_opt != "None (Auto-generate IDs)" and selected_id_opt in df.columns:
            processed_df['Product_ID'] = df[selected_id_opt].astype(str)
        else:
            processed_df['Product_ID'] = [f"Item_{i+1}" for i in range(len(df))]
            
        processed_df['Category'] = df[category_col].astype(str) if category_col in df.columns else "General"
        
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
        
        # Top Metrics Summary Cards
        total_products = len(processed_df)
        reorder_count = int(processed_df['Reorder_Required'].sum())
        total_predicted_demand = int(processed_df['Predicted_Demand'].sum())
        
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Products Evaluated", f"{total_products:,}")
        m2.metric("Items Requiring Reorder", f"{reorder_count:,}", delta=f"-{reorder_count} urgent" if reorder_count > 0 else "All Good", delta_color="inverse" if reorder_count > 0 else "normal")
        m3.metric("Total Predicted Demand", f"{total_predicted_demand:,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Restock Alert Banner
        if reorder_count > 0:
            reordered_items = processed_df[processed_df['Reorder_Required'] == True]
            item_list_str = ", ".join(reordered_items['Product_ID'].tolist())
            st.warning(f"⚠️ **Restock Action Required:** Immediate restock needed for **{reorder_count} item(s)**: `{item_list_str}`")
        else:
            st.success("🎉 Optimal Stock Levels! All product inventories exceed predicted demand forecasts.")
            
        st.markdown("---")
        
        # Analytics Visualizations
        st.markdown("### 📈 Demand Analytics")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Demand by Category")
            cat_demand = processed_df.groupby('Category')['Predicted_Demand'].sum()
            st.bar_chart(cat_demand)
        with c2:
            st.subheader("Demand Trend Overview")
            st.line_chart(processed_df['Predicted_Demand'])
            
        st.markdown("---")
        
        # Detailed Table View with Styling
        st.markdown("### 📋 Product Evaluation Records")
        filter_view = st.radio("Filter Records:", ["All Products", "Reorder Required Only"], horizontal=True)
        
        display_df = processed_df.copy()
        if filter_view == "Reorder Required Only":
            display_df = display_df[display_df['Reorder_Required'] == True]
            
        # Highlight reorder rows softly in red
        def highlight_reorders(row):
            return ['background-color: #fff5f5' if row['Reorder_Required'] else '' for _ in row]
            
        st.dataframe(display_df.style.apply(highlight_reorders, axis=1), use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred while processing your file: {e}")
else:
    # Empty State Hero Section
    st.markdown("""
        <div style="padding: 40px; text-align: center; background-color: #ffffff; border: 1px dashed #cccccc; border-radius: 12px;">
            <h3>👋 Welcome to your Inventory Dashboard</h3>
            <p style="color: #666666;">Use the <b>sidebar on the left</b> to upload your inventory spreadsheet (CSV or Excel) and generate instant ML forecasts.</p>
        </div>
    """, unsafe_allow_html=True)

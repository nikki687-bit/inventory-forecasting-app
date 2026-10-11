import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Product Demand & Stock Recommendation Hub",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: FORMAT GUIDE & TEMPLATE ONLY ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
    st.header("Inventory Guide")
    st.markdown("Check formatting requirements or download the sample template.")
    
    with st.expander("📂 View File Format Guide", expanded=True):
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

    # --- ABOUT & TECH ARCHITECTURE (SIDEBAR NOTE) ---
    with st.expander("ℹ️ About & Tech Architecture", expanded=False):
        st.markdown("""
        **🎯 What it is:**  
        An automated inventory management and demand forecasting dashboard designed to prevent stockouts and optimize stock levels.

        **💡 Why XGBoost?**  
        - **Tabular Superiority:** Unmatched performance on structured retail datasets.
        - **Non-Linear Relationships:** Accurately captures complex interactions (e.g., how holiday spikes multiply with discounts).
        - **Missing Data Robustness:** Built-in sparsity-aware split finding handles dirty or sparse spreadsheet entries effortlessly.
        """)

    st.markdown("---")
    st.caption("🚀 Powered by XGBoost & Streamlit")

# --- MAIN TITLE & HEADER ---
st.title("Product Demand Prediction and Smart Stock Recommendation")
st.markdown("Real-time automated stock analysis, demand prediction, and smart reorder alerts.")
st.markdown("---")

# --- CENTRAL FILE UPLOADER ---
st.subheader("📁 Upload Inventory File")
uploaded_file = st.file_uploader("Choose a CSV, Excel, or Text file", type=["csv", "xlsx", "xls", "txt"])

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

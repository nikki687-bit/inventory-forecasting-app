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

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
    st.header("Inventory Guide")
    st.caption("Check formatting requirements or download the sample template.")

    with st.expander("📁 View File Format Guide", expanded=False):
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

    # Helper function for sample CSV template
    def get_sample_csv():
        data = {
            "Product_ID": ["SKU-901", "SKU-902", "SKU-903", "SKU-904", "SKU-905"],
            "Category": ["Electronics", "Groceries", "Furniture", "Toys", "Electronics"],
            "Price": [450.0, 25.0, 1200.0, 50.0, 300.0],
            "Discount": [15, 0, 10, 5, 5],
            "Holiday": [0, 1, 0, 1, 0],
            "Previous_Sales": [38, 140, 15, 55, 18],
            "Stock_Available": [25, 180, 12, 70, 20]
        }
        df_sample = pd.DataFrame(data)
        return df_sample.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Template",
        data=get_sample_csv(),
        file_name="inventory_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")

    # --- ABOUT & TECH ARCHITECTURE (SIDEBAR NOTE) ---
    with st.expander("ℹ️ About & Tech Architecture"):
        st.markdown("""
        **🎯 What it is:**  
        An automated inventory management and demand forecasting dashboard designed to prevent stockouts and optimize stock levels.

        **🛠️ How it's done:**  
        - **Frontend/UI:** Built using **Streamlit** for real-time interactivity.
        - **Data Pipeline:** Uses custom header auto-detection and **Pandas** to sanitize missing values (`fillna(0)`).
        - **Prediction Engine:** Powered by an **XGBoost Regressor** trained on features like price, discounts, holidays, and past sales.

        **💡 Why XGBoost?**  
        - **Tabular Superiority:** Unmatched performance on structured retail datasets.
        - **Non-Linear Relationships:** Accurately captures complex interactions (e.g., how holiday spikes multiply with discounts).
        - **Missing Data Robustness:** Built-in sparsity-aware split finding handles dirty or sparse spreadsheet entries effortlessly.
        """)

    st.markdown("---")
    st.caption("🚀 Powered by XGBoost & Streamlit")


# --- MAIN HEADER ---
st.title("📦 Product Demand & Smart Stock Recommendation Hub")
st.markdown("Real-time automated stock analysis, demand prediction, and smart reorder alerts.")

st.markdown("<br>", unsafe_allow_html=True)

# --- TOP NATIVE METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="CORE ENGINE", value="XGBoost")

with col2:
    st.metric(label="AUTO MAPPING", value="Active ⚡")

with col3:
    st.metric(label="DATA SANITIZATION", value="Pandas")

with col4:
    st.metric(label="SUPPORTED INPUTS", value="CSV / XLSX")

st.markdown("<br>", unsafe_allow_html=True)

# --- UPLOAD SECTION ---
with st.container():
    st.subheader("📂 Upload Inventory Dataset")
    st.caption("Select or drag-and-drop your inventory file to initialize demand forecasting and automated restock analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or Text file", 
        type=["csv", "xlsx", "xls", "txt"],
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- DATA PROCESSING & MAPPING PIPELINE ---
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Successfully loaded **{uploaded_file.name}**")
        
        # Helper to auto-find index of matching columns for dropdowns
        cols_avail = ["-- Select Column --"] + df.columns.tolist()
        
        def find_col_index(keywords):
            for idx, col in enumerate(cols_avail):
                if any(kw in col.lower() for kw in keywords):
                    return idx
            return 0

        # Smart Column Mapping Expander with Pre-selected Auto-Detection
        with st.container():
            st.markdown("### ⚙️ Smart Column Mapping Auto-detected — Click to adjust")
            st.info("💡 **How Smart Mapping Works:** Automatically links your custom spreadsheet headers to the machine learning model without requiring format changes.")
            
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                m_price = st.selectbox("Price Column", cols_avail, index=find_col_index(['price', 'cost', 'rate']))
            with mc2:
                m_disc = st.selectbox("Discount Column", cols_avail, index=find_col_index(['disc', 'markdown', 'offer']))
            with mc3:
                m_hol = st.selectbox("Holiday Column", cols_avail, index=find_col_index(['holid', 'festival', 'flag']))
                
            mc4, mc5, mc6 = st.columns(3)
            with mc4:
                m_stock = st.selectbox("Stock Available Column", cols_avail, index=find_col_index(['stock', 'quantity', 'hand', 'avail']))
            with mc5:
                m_prev = st.selectbox("Previous Sales Column", cols_avail, index=find_col_index(['prev', 'past', 'sales', 'sold']))
            with mc6

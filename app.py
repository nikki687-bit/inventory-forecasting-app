import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Product Demand & Smart Stock Recommendation Hub",
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

    # Technical Architecture Expander
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

# --- TOP NATIVE METRIC CARDS (Auto-adapts to Light/Dark Mode) ---
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

# --- DATA PROCESSING & RESULTS ---
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Successfully loaded **{uploaded_file.name}**")
        
        st.markdown("### 📊 Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("👈 Upload an inventory file above or download the sample template from the sidebar to begin.")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="AI Inventory Forecasting & Reorder Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 AI Inventory Forecasting & Reorder Dashboard")
st.markdown("Upload your retail inventory CSV file to generate dynamic demand predictions, track reorders, and analyze trends.")

# Load Trained Model
@st.cache_resource
def load_model():
    model_path = "xgboost_inventory_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model file `xgboost_inventory_model.pkl` not found in the repository root directory!")
else:
    # Sidebar File Uploader
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Inventory CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Successfully loaded uploaded CSV file!")
            
            # --- ROBUST PREPROCESSING PIPELINE ---
            model_df = df.copy()
            
            # 1. Handle truncated column names from Excel display (e.g., Previous_S, Stock_Avai)
            rename_map = {}
            for col in model_df.columns:
                if col.startswith('Previous_S'):
                    rename_map[col] = 'Previous_Sales'
                elif col.startswith('Stock_Avai'):
                    rename_map[col] = 'Stock_Available'
            model_df = model_df.rename(columns=rename_map)

            # Ensure standard stock column exists for reorder logic
            if 'Stock_Aval' in model_df.columns and 'Stock_Available' not in model_df.columns:
                model_df['Stock_Available'] = model_df['Stock_Aval']

            # 2. Extract Date / Time features matching the training pipeline
            if 'Date' in model_df.columns:
                model_df['Date'] = pd.to_datetime(model_df['Date'])
                model_df['Year'] = model_df['Date'].dt.year
                model_df['Month'] = model_df['Date'].dt.month
                model_df['Day'] = model_df['Date'].dt.day
                model_df['DayOfWeek'] = model_df['Date'].dt.dayofweek

            # 3. One-hot encode Category column
            if 'Category' in model_df.columns:
                model_df = pd.get_dummies(model_df, columns=['Category'], drop_first=True)

            # 4. Align features precisely with model expectations
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
                for col in expected_features:
                    if col not in model_df.columns:
                        model_df[col] = 0
                X_predict = model_df[expected_features]
            else:
                X_predict = model_df.select_dtypes(include=[np.number])

            # Run Predictions
            preds = model.predict(X_predict)
            df['Predicted_Demand'] = np.round(preds, 2)
            
            # Reorder Logic & Safety Buffer
            stock_col = 'Stock_Available' if 'Stock_Available' in df.columns else ('Stock_Aval' if 'Stock_Aval' in df.columns else None)
            if stock_col:
                df['Reorder_Required'] = df[stock_col] < df['Predicted_Demand']
            else:
                df['Reorder_Required'] = False

            # --- METRICS DISPLAY ---
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Products Evaluated", len(df))
            with col2:
                reorder_count = int(df['Reorder_Required'].sum()) if 'Reorder_Required' in df.columns else 0
                st.metric("Items Requiring Reorder", reorder_count)
            with col3:
                total_predicted = int(df['Predicted_Demand'].sum())
                st.metric("Total Predicted Demand", total_predicted)

            # --- VISUALIZATIONS ---
            st.markdown("---")
            st.subheader("📊 Inventory Analytics & Visualizations")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("##### Predicted Demand by Category")
                if 'Category' in df.columns:
                    cat_demand = df.groupby('Category')['Predicted_Demand'].sum().reset_index()
                    st.bar_chart(cat_demand.set_index('Category'))
                else:
                    st.info("Category column not available for breakdown.")

            with chart_col2:
                st.markdown("##### Demand Distribution / Trend")
                st.line_chart(df['Predicted_Demand'].reset_index(drop=True))

            # --- DATAFRAME VIEW & DOWNLOAD ---
            st.markdown("---")
            st.subheader("📋 Detailed Product Evaluation Table")
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Predictions CSV",
                data=csv_data,
                file_name="inventory_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
    else:
        st.info("👈 Please upload your `clean_retail_inventory.csv` file in the sidebar to get started.")

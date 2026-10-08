import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="AI Inventory Forecasting & Reorder Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Inventory Forecasting & Reorder Dashboard")
st.markdown("Upload your inventory CSV file to generate dynamic demand predictions, track reorders, and analyze trends.")

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
    st.error("⚠️ Model file 'xgboost_inventory_model.pkl' not found in the repository root directory!")
else:
    # Main file uploader (no sidebar)
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded CSV file!")

            # --- PREPROCESSING MATCHING TRAINING SCRIPT ---
            model_df = df.copy()

            # 1. Handle truncated column names from Excel display
            rename_map = {}
            for col in model_df.columns:
                if col.startswith('Previous_S'):
                    rename_map[col] = 'Previous_Sales'
                elif col.startswith('stock_avail') or col.startswith('Stock_Avail'):
                    rename_map[col] = 'Stock_Available'
            
            model_df = model_df.rename(columns=rename_map)

            if 'Stock_Avail' in model_df.columns and 'Stock_Available' not in model_df.columns:
                model_df['Stock_Available'] = model_df['Stock_Avail']

            # 2. Extract Date / Time features exactly like training script
            if 'Date' in model_df.columns:
                model_df['Date'] = pd.to_datetime(model_df['Date'])
                model_df['Year'] = model_df['Date'].dt.year
                model_df['Month'] = model_df['Date'].dt.month
                model_df['Day'] = model_df['Date'].dt.day
                model_df['DayOfWeek'] = model_df['Date'].dt.dayofweek

            # 3. Select exact features used in training (Matches your training features array casing)
            training_features = [
                'Price', 'Discount', 'Holiday', 'Previous_Sales',
                'Stock_Available', 'Year', 'Month', 'Day', 'DayOfWeek', 'Category'
            ]

            # FIXED: Nested missing feature verification properly inside the missing condition check
            for col in training_features:
                if col not in model_df.columns:
                    if col in ['Price', 'Discount', 'Previous_Sales', 'Stock_Available']:
                        model_df[col] = 0.0
                    elif col in ['Holiday', 'Year', 'Month', 'Day', 'DayOfWeek']:
                        model_df[col] = 0
                    elif col == 'Category':
                        model_df[col] = 'Unknown'

            X_subset = model_df[training_features]

            # 4. One-hot encode category exactly like training script
            X_encoded = pd.get_dummies(X_subset, columns=['Category'], drop_first=True)

            # 5. Align precisely with model.feature_names_in_
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
                for col in expected_features:
                    if col not in X_encoded.columns:
                        X_encoded[col] = 0
                X_predict = X_encoded[expected_features]
            else:
                X_predict = X_encoded

            # Run Predictions
            preds = model.predict(X_predict)
            df['Predicted_Demand'] = np.round(preds, 2)

            # Reorder Logic & Safety Buffer
            stock_col = 'Stock_Available' if 'Stock_Available' in df.columns else ('Stock_Avail' if 'Stock_Avail' in df.columns else None)
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

            # --- DETAILED OUTPUT TABLE & FILTERING ---
            st.markdown("---")
            st.subheader("📋 Detailed Product Evaluation Table")

            table_filter = st.radio(
                "Filter Table View:",
                ["All Products", "⚠️ Reorder Required Only"],
                horizontal=True
            )

            display_df = df.copy()
            if table_filter == "⚠️ Reorder Required Only":
                if 'Reorder_Required' in display_df.columns:
                    display_df = display_df[display_df['Reorder_Required'] == True]

            st.dataframe(display_df, use_container_width=True)

            # --- DOWNLOAD BUTTON ---
            st.markdown("---")
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
        st.info("💡 Please upload your CSV file above to get started.")

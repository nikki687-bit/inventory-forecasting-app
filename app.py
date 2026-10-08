import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Inventory Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

# App Title & Description
st.title("📦 AI Inventory Demand Forecasting & Reorder System")
st.markdown("Upload your batch inventory CSV file to generate predictive demand analytics, safety buffers, and automated reorder recommendations.")

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
    st.error("Model file 'xgboost_inventory_model.pkl' not found in the repository. Please ensure it is uploaded.")
else:
    # File uploader widget
    uploaded_file = st.file_uploader("Upload Batch Inventory CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded CSV file!")
            
            # Preprocessing matching the loaded model's expected features
            model_df = df.copy()
            
            # Handle categorical 'Category' column if present as raw text
            if 'Category' in model_df.columns:
                model_df = pd.get_dummies(model_df, columns=['Category'], drop_first=True)
            
            # Align features with model expectations
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
            
            # Calculate Safety Buffer & Reorder Recommendations
            df['Safety_Buffer'] = np.round(df['Predicted_Demand'] * 0.15, 2)
            df['Recommended_Reorder'] = np.maximum(0, np.ceil((df['Predicted_Demand'] + df['Safety_Buffer']) - df['Stock_Available'])) if 'Stock_Available' in df.columns else np.maximum(0, np.ceil((df['Predicted_Demand'] + df['Safety_Buffer']) - df['Current_Stock']))
            
            stock_col = 'Stock_Available' if 'Stock_Available' in df.columns else 'Current_Stock'
            df['Status'] = np.where(df[stock_col] < (df['Predicted_Demand'] + df['Safety_Buffer']), 'Action Required: Reorder', 'Stock Sufficient')

            # Metrics Summary Section
            total_products = len(df)
            items_reorder = len(df[df['Status'] == 'Action Required: Reorder'])
            total_demand = int(df['Predicted_Demand'].sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Products Evaluated", f"{total_products:,}")
            col2.metric("Items Requiring Reorder", f"{items_reorder:,}")
            col3.metric("Total Predicted Demand", f"{total_demand:,}")

            st.markdown("---")

            # Visualizations Section (Native Streamlit Bar & Line Charts)
            st.subheader("📊 Inventory Analytics & Visualizations")

            g_col1, g_col2 = st.columns(2)

            with g_col1:
                st.markdown("**Predicted Demand by Category (Bar Chart)**")
                # Find category columns or fallback to grouping
                cat_cols = [c for c in df.columns if 'Category' in c]
                if cat_cols:
                    # Melt or sum by category
                    # If columns are one-hot encoded:
                    cat_summary = {}
                    for col in cat_cols:
                        cat_name = col.replace('Category_', '')
                        cat_summary[cat_name] = df[df[col] == 1]['Predicted_Demand'].sum()
                    st.bar_chart(pd.Series(cat_summary))
                elif 'Category' in df.columns:
                    cat_demand = df.groupby('Category')['Predicted_Demand'].sum()
                    st.bar_chart(cat_demand)
                else:
                    st.bar_chart(df['Predicted_Demand'])

            with g_col2:
                st.markdown("**Demand Distribution / Trend**")
                if 'Date' in df.columns:
                    date_trend = df.groupby('Date')['Predicted_Demand'].sum()
                    st.line_chart(date_trend)
                else:
                    st.line_chart(df['Predicted_Demand'])

            st.markdown("---")

            # Detailed Output Table & Batch CSV Download
            st.subheader("📋 Detailed Output Table")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Prediction Results (CSV)",
                data=csv_data,
                file_name="batch_inventory_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

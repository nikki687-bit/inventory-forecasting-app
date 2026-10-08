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
            
            # Verify required columns exist
            required_cols = ['Price', 'Discount', 'Holiday', 'Previous_Sales', 'Stock_Available', 'Category']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"The uploaded CSV is missing these required feature columns: {missing_cols}")
            else:
                # Preprocessing & Feature Engineering matching training
                model_df = df.copy()
                
                # Convert Date to datetime and extract date components matching training
                if 'Date' in model_df.columns:
                    model_df['Date'] = pd.to_datetime(model_df['Date'])
                    model_df['Year'] = model_df['Date'].dt.year
                    model_df['Month'] = model_df['Date'].dt.month
                    model_df['Day'] = model_df['Date'].dt.day
                    model_df['DayOfWeek'] = model_df['Date'].dt.dayofweek
                
                # One-hot encode Category
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
                df['Recommended_Reorder'] = np.maximum(0, np.ceil((df['Predicted_Demand'] + df['Safety_Buffer']) - df['Stock_Available']))
                df['Status'] = np.where(df['Stock_Available'] < (df['Predicted_Demand'] + df['Safety_Buffer']), 'Action Required: Reorder', 'Stock Sufficient')

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
                    if 'Category' in df.columns:
                        cat_demand = df.groupby('Category')['Predicted_Demand'].sum()
                        st.bar_chart(cat_demand)
                    else:
                        st.info("Category column not found for plotting.")

                with g_col2:
                    st.markdown("**Demand Trend Over Time (Line Chart)**")
                    if 'Date' in df.columns:
                        date_trend = df.groupby('Date')['Predicted_Demand'].sum()
                        st.line_chart(date_trend)
                    else:
                        st.info("Date column not found for trend line chart.")

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

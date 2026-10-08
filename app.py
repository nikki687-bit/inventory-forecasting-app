import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title="Inventory Demand Forecasting Dashboard", layout="wide")

st.title("📦 Inventory Demand & Reorder Forecasting Dashboard")
st.markdown("Upload your inventory CSV file (with columns like `Date`, `Product_ID`, `Category`, `Price`, `Discount`, `Holiday`, `Previous_Sales`, `Stock_Available`) to generate demand predictions and reorder recommendations.")

# Load trained XGBoost model safely
@st.cache_resource
def load_model():
    return pickle.load(open('xgboost_inventory_model.pkl', 'rb'))

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader widget (Waits exclusively for user file)
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

            # Display Metrics Summary
            total_products = len(df)
            items_reorder = len(df[df['Status'] == 'Action Required: Reorder'])
            total_demand = int(df['Predicted_Demand'].sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Products Evaluated", f"{total_products:,}")
            col2.metric("Items Requiring Reorder", f"{items_reorder:,}")
            col3.metric("Total Predicted Demand", f"{total_demand:,}")

            st.markdown("---")

            # --- VISUALIZATIONS SECTION (Native Streamlit Bar & Line Charts) ---
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

            # Detailed Output Table
            st.subheader("📋 Detailed Output Table")
            st.dataframe(df, use_container_width=True)

            # Download Button for Results
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Predictions CSV",
                data=csv_output,
                file_name="inventory_forecast_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("👆 Please upload your inventory CSV file above to begin forecasting.")

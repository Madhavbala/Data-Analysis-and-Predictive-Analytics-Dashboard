import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set the style for Seaborn plots
sns.set_style('whitegrid')

# Title of the application
st.title("Data Analysis and Prediction Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page", ["Home", "Data Overview", "Visualizations", "Predictive Analytics"])

# Load datasets
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

if 'data' not in st.session_state:
    st.session_state['data'] = None

if page == "Home":
    st.header("Welcome to the Data Dashboard")
    st.write("""
    This application allows you to upload an dataset, analyze key metrics, 
    and visualize data trends. You can also perform predictive analytics to estimate charges 
    based on the available features.
    """)

    st.subheader("Project Purpose")
    st.write("""
    The purpose of this project is to empower users to gain insights from insurance-related data through 
    comprehensive analysis and visualization tools. By providing a user-friendly interface for data 
    exploration, the application enables users to understand patterns and relationships within their 
    data, ultimately facilitating informed decision-making in the insurance domain.
    """)


elif page == "Data Overview":
    st.header("Data Overview")
    uploaded_file = st.file_uploader("Upload your insurance CSV file", type=["csv"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state['data'] = df
        
        st.subheader(f"Data Overview")
        st.dataframe(df.head())

        st.subheader("Data Shape")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        st.subheader("Descriptive Statistics")
        st.write(df.describe())
    else:
        st.info("Please upload a CSV file to get started.")

elif page == "Visualizations":
    st.header("Data Visualizations")
    
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        
        columns = df.columns.tolist()
        x_column = st.selectbox("Select X-axis column", columns)
        y_column = st.selectbox("Select Y-axis column", [None] + columns)
        plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Line Plot", "Bar Plot", 
                                                      "Histogram", "Box Plot", "Violin Plot", 
                                                      "Correlation Heatmap"])
        
        if st.button("Generate Plot"):
            plt.figure(figsize=(12, 8), dpi=200)
            
            if plot_type == "Scatter Plot":
                sns.scatterplot(x=x_column, y=y_column, data=df)
                plt.title(f'Scatter Plot of {y_column} vs {x_column}', fontsize=16)
                plt.xlabel(x_column, fontsize=14)
                plt.ylabel(y_column, fontsize=14)

            elif plot_type == "Line Plot":
                sns.lineplot(x=x_column, y=y_column, data=df)
                plt.title(f'Line Plot of {y_column} over {x_column}', fontsize=16)
                plt.xlabel(x_column, fontsize=14)
                plt.ylabel(y_column, fontsize=14)

            elif plot_type == "Bar Plot":
                sns.barplot(x=x_column, y=y_column, data=df)
                plt.title(f'Bar Plot of {y_column} by {x_column}', fontsize=16)
                plt.xlabel(x_column, fontsize=14)
                plt.ylabel(y_column, fontsize=14)

            elif plot_type == "Histogram":
                sns.histplot(df[x_column], kde=True)
                plt.title(f'Histogram of {x_column}', fontsize=16)
                plt.xlabel(x_column, fontsize=14)
                plt.ylabel('Frequency', fontsize=14)

            elif plot_type == "Box Plot":
                sns.boxplot(x=x_column, y=y_column, data=df)
                plt.title(f'Box Plot of {y_column} by {x_column}', fontsize=16)
                plt.xlabel(x_column, fontsize=14)
                plt.ylabel(y_column, fontsize=14)

            elif plot_type == "Violin Plot":
                sns.violinplot(x=x_column, y=y_column, data=df)
                plt.title(f'Violin Plot of {y_column} by {x_column}', fontsize=16)
                plt.xlabel(x_column, fontsize=14)
                plt.ylabel(y_column, fontsize=14)

            elif plot_type == "Correlation Heatmap":
                corr = df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title('Correlation Heatmap', fontsize=16)
                
            st.pyplot(plt)
            
    else:
        st.warning("Please upload at least one dataset first.")
elif page == "Predictive Analytics":
    st.header("Predictive Analytics")

    # Check if data exists in session state
    if st.session_state.get('data') is not None:
        df = st.session_state['data']

        # Convert 'Date' to datetime if it's not already
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

        # Allow users to select any numeric column as the target variable for prediction
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        target = st.selectbox("Select target variable for prediction", numeric_columns)

        # Select features to use for prediction
        features = st.multiselect(
            "Select features for prediction",
            df.columns.tolist(),  # Show all columns
            default=[col for col in df.columns if col != target]  # Default to all except target
        )

        if target and features:
            # Handle date feature: Extract relevant information
            if 'Date' in features:
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                features.remove('Date')
                features.extend(['Year', 'Month', 'Day'])

            # Separate features into numeric and categorical types
            numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = df[features].select_dtypes(include=['object']).columns.tolist()

            # Preparing the data
            X = df[features]
            y = df[target]

            # Apply scaling to numeric features and one-hot encoding to categorical features
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline

            # Create a pipeline that scales numeric data and one-hot encodes categorical data
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model selection dropdown
            model_choice = st.selectbox("Select a model for prediction", ["Linear Regression", "Random Forest", "Support Vector Regression (SVR)"])

            # Initialize the chosen model
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_choice == "Support Vector Regression (SVR)":
                model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

            # Create a pipeline that applies preprocessing and then the chosen model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            # Predict on test set
            predictions = pipeline.predict(X_test)

            # Calculate and display R-squared score for the model's accuracy
            accuracy = r2_score(y_test, predictions)
            st.write(f"Model Accuracy (R-squared): {accuracy:.2f}")

            # Allow user to input new data for prediction
            st.subheader("Predict New Data")
            new_data = {}
            for feature in features:
                if feature in numeric_features:
                    new_data[feature] = st.number_input(f"Input {feature}", value=float(df[feature].mean()))
                else:
                    new_data[feature] = st.selectbox(f"Input {feature}", df[feature].unique())

            # Convert new data to DataFrame
            new_df = pd.DataFrame(new_data, index=[0])

            # Predict with the trained model
            predicted_value = pipeline.predict(new_df)[0]
            st.write(f"Predicted {target}: {predicted_value:.2f}")

            # Color-coded indicator for increase or decrease in predicted value
            last_actual_value = y.iloc[-1]  # Last actual value for the target
            if predicted_value > last_actual_value:
                st.markdown(f"<p style='color:green;'>Increase: {predicted_value:.2f}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:red;'>Decrease: {predicted_value:.2f}</p>", unsafe_allow_html=True)

        else:
            st.warning("Please select both a target and at least one feature for prediction.")

    else:
        st.warning("Please upload a dataset first.")

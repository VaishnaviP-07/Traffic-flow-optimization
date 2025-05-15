import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def train_traffic_prediction_model(traffic_df):
    """
    Train a machine learning model to predict traffic volume
    
    Args:
        traffic_df: Traffic data dataframe
        
    Returns:
        Trained model and preprocessing components
    """
    if traffic_df.empty:
        return None, None, None
    
    # Feature engineering
    df = traffic_df.copy()
    
    # Extract datetime features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
    
    # Prepare features and target
    X = df[['hour', 'day_of_week', 'month', 'is_weekend']]
    categorical_cols = ['location']
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Combine features
    X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    y = df['volume']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }
    
    return model, scaler, encoder, metrics

def predict_traffic_volume(model, scaler, encoder, location, hour, day_of_week, month, is_weekend):
    """
    Predict traffic volume for specific conditions
    
    Args:
        model: Trained ML model
        scaler: Fitted StandardScaler
        encoder: Fitted OneHotEncoder
        location, hour, day_of_week, month, is_weekend: Prediction features
        
    Returns:
        Predicted traffic volume
    """
    if model is None or scaler is None or encoder is None:
        return 0
    
    # Create feature vector
    features = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'is_weekend': [is_weekend]
    })
    
    # Encode location
    location_df = pd.DataFrame({'location': [location]})
    encoded_location = encoder.transform(location_df)
    encoded_location_df = pd.DataFrame(encoded_location, columns=encoder.get_feature_names_out(['location']))
    
    # Combine features
    X = pd.concat([features.reset_index(drop=True), encoded_location_df.reset_index(drop=True)], axis=1)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return prediction

def evaluate_model_performance(model, X_test, y_test, scaler):
    """
    Evaluate model performance with visualizations
    
    Args:
        model: Trained ML model
        X_test: Test features
        y_test: Test target values
        scaler: Fitted StandardScaler
        
    Returns:
        Dictionary of metrics and Plotly figure
    """
    if model is None:
        return {}, px.scatter()
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }
    
    # Create comparison plot
    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    
    fig = px.scatter(
        comparison_df, 
        x='Actual', 
        y='Predicted',
        title='Model Performance: Actual vs. Predicted Traffic Volume',
        labels={'Actual': 'Actual Volume', 'Predicted': 'Predicted Volume'}
    )
    
    # Add perfect prediction line
    max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
    min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    return metrics, fig

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the model
    
    Args:
        model: Trained ML model
        feature_names: Names of features
        
    Returns:
        Plotly figure showing feature importance
    """
    if model is None or not hasattr(model, 'feature_importances_'):
        return px.bar()
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        importance_df, 
        x='Feature', 
        y='Importance',
        title='Feature Importance',
        labels={'Feature': 'Feature', 'Importance': 'Importance'}
    )
    
    return fig

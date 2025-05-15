import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

def prepare_prophet_data(df):
    """
    Prepare data for Prophet time series model
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Dataframe formatted for Prophet
    """
    if df.empty:
        return pd.DataFrame()
    
    # Prophet requires columns named 'ds' and 'y'
    prophet_df = df.copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df['timestamp'])
    prophet_df['y'] = prophet_df['volume']
    
    # Group by timestamp to avoid duplicates
    prophet_df = prophet_df.groupby('ds').agg({'y': 'mean'}).reset_index()
    
    return prophet_df

def predict_traffic(df, prediction_days=7):
    """
    Predict future traffic using Prophet time series model
    
    Args:
        df: Traffic data dataframe
        prediction_days: Number of days to predict
        
    Returns:
        Forecast dataframe and plot
    """
    if df.empty:
        return pd.DataFrame(), px.line()
    
    # Prepare data for Prophet
    prophet_df = prepare_prophet_data(df)
    
    if prophet_df.empty:
        return pd.DataFrame(), px.line()
    
    # Create and fit model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,  # Set to True if data spans multiple years
        seasonality_mode='multiplicative'
    )
    
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=prediction_days*24, freq='H')
    
    # Make prediction
    forecast = model.predict(future)
    
    # Create forecast plot
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            mode='markers',
            name='Actual',
            marker=dict(color='blue', size=4)
        )
    )
    
    # Add prediction line
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        )
    )
    
    # Add prediction interval
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty',
            showlegend=False
        )
    )
    
    # Adjust layout
    fig.update_layout(
        title='Traffic Volume Forecast',
        xaxis_title='Date',
        yaxis_title='Traffic Volume',
        hovermode='x unified'
    )
    
    return forecast, fig

def identify_peak_times(forecast_df):
    """
    Identify peak traffic times from forecast
    
    Args:
        forecast_df: Prophet forecast dataframe
        
    Returns:
        Dictionary with peak times information
    """
    if forecast_df.empty:
        return {}
    
    # Add day of week and hour columns
    forecast_df['day_of_week'] = forecast_df['ds'].dt.day_name()
    forecast_df['hour'] = forecast_df['ds'].dt.hour
    
    # Find peak hours for each day
    peak_hours = {}
    
    for day in forecast_df['day_of_week'].unique():
        day_data = forecast_df[forecast_df['day_of_week'] == day]
        
        # Morning peak (5-12)
        morning_data = day_data[(day_data['hour'] >= 5) & (day_data['hour'] <= 12)]
        if not morning_data.empty:
            morning_peak_hour = morning_data.loc[morning_data['yhat'].idxmax()]['hour']
            morning_peak_volume = morning_data['yhat'].max()
        else:
            morning_peak_hour = None
            morning_peak_volume = 0
        
        # Evening peak (13-20)
        evening_data = day_data[(day_data['hour'] >= 13) & (day_data['hour'] <= 20)]
        if not evening_data.empty:
            evening_peak_hour = evening_data.loc[evening_data['yhat'].idxmax()]['hour']
            evening_peak_volume = evening_data['yhat'].max()
        else:
            evening_peak_hour = None
            evening_peak_volume = 0
        
        peak_hours[day] = {
            'morning_peak': f"{int(morning_peak_hour)}:00" if morning_peak_hour is not None else "N/A",
            'morning_volume': int(morning_peak_volume),
            'evening_peak': f"{int(evening_peak_hour)}:00" if evening_peak_hour is not None else "N/A",
            'evening_volume': int(evening_peak_volume)
        }
    
    return peak_hours

def decompose_time_series(df):
    """
    Decompose time series into trend, seasonality, and residual components
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Dictionary of component plots
    """
    if df.empty:
        return {}
    
    # Prepare data for Prophet
    prophet_df = prepare_prophet_data(df)
    
    if prophet_df.empty or len(prophet_df) < 2:
        return {}
    
    # Create and fit model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,  # Set to True if data spans multiple years
        seasonality_mode='multiplicative'
    )
    
    model.fit(prophet_df)
    
    # Predict on historical data
    forecast = model.predict(prophet_df)
    
    # Create trend component plot
    trend_fig = px.line(
        forecast, 
        x='ds', 
        y='trend',
        title='Traffic Volume Trend Component',
        labels={'ds': 'Date', 'trend': 'Trend'}
    )
    
    # Create daily seasonality plot
    days = pd.date_range(start='2023-01-01', periods=24, freq='H')
    daily_seasonality = model.predict_seasonal_components(pd.DataFrame({'ds': days}))
    
    daily_fig = px.line(
        daily_seasonality, 
        x=daily_seasonality.index, 
        y='daily',
        title='Daily Seasonality Component',
        labels={'index': 'Hour of Day', 'daily': 'Effect'}
    )
    daily_fig.update_xaxes(tickvals=list(range(0, 24)))
    
    # Create weekly seasonality plot
    days = pd.date_range(start='2023-01-01', periods=7, freq='D')
    weekly_seasonality = model.predict_seasonal_components(pd.DataFrame({'ds': days}))
    
    weekly_fig = px.line(
        weekly_seasonality, 
        x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
        y='weekly',
        title='Weekly Seasonality Component',
        labels={'x': 'Day of Week', 'weekly': 'Effect'}
    )
    
    return {
        'trend': trend_fig,
        'daily': daily_fig,
        'weekly': weekly_fig
    }

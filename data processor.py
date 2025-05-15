import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def filter_data_by_date(df, start_date, end_date):
    """
    Filter traffic data based on date range
    
    Args:
        df: Traffic data dataframe
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Filtered dataframe
    """
    if df.empty:
        return df
    
    # Convert timestamps to datetime if they are strings
    if isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert start_date and end_date to datetime for comparison
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Filter data
    filtered_data = df[(df['timestamp'] >= start_datetime) & 
                       (df['timestamp'] <= end_datetime)]
    
    return filtered_data

def calculate_signal_efficiency(signals_df, traffic_df):
    """
    Calculate overall signal efficiency based on signal data and traffic volume
    
    Args:
        signals_df: Traffic signals dataframe
        traffic_df: Traffic volume dataframe
        
    Returns:
        Overall efficiency percentage
    """
    if signals_df.empty:
        return 0
    
    # Use the average efficiency from signals data
    return signals_df['efficiency'].mean()

def plot_traffic_by_time(df):
    """
    Create a time series plot of traffic volume
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Plotly figure object
    """
    # Group by hourly timestamp and calculate mean volume
    df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
    hourly_data = df.groupby('hour').agg({'volume': 'mean'}).reset_index()
    
    # Create time series plot
    fig = px.line(
        hourly_data, 
        x='hour', 
        y='volume',
        title='Traffic Volume Over Time',
        labels={'hour': 'Date & Time', 'volume': 'Traffic Volume'}
    )
    fig.update_layout(hovermode='x unified')
    
    return fig

def plot_traffic_by_location(df):
    """
    Create a bar chart of traffic volume by location
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Plotly figure object
    """
    # Group by location and calculate mean volume
    location_data = df.groupby('location').agg({'volume': 'mean'}).reset_index()
    location_data = location_data.sort_values('volume', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        location_data, 
        x='location', 
        y='volume',
        title='Average Traffic Volume by Location',
        labels={'location': 'Location', 'volume': 'Average Traffic Volume'}
    )
    
    return fig

def get_signal_statuses(df):
    """
    Get counts of signal statuses
    
    Args:
        df: Traffic signals dataframe
        
    Returns:
        Dictionary with status counts
    """
    if df.empty:
        return {'operational': 0, 'issues': 0}
    
    statuses = df['status'].value_counts().to_dict()
    return {
        'operational': statuses.get('Operational', 0),
        'issues': statuses.get('Issue', 0)
    }

def plot_signal_efficiency(signals_df, traffic_df):
    """
    Create a bar chart of signal efficiency by location
    
    Args:
        signals_df: Traffic signals dataframe
        traffic_df: Traffic volume dataframe
        
    Returns:
        Plotly figure object
    """
    if signals_df.empty:
        return px.bar()
    
    # Group by location and calculate mean efficiency
    efficiency_data = signals_df.groupby('location').agg({'efficiency': 'mean'}).reset_index()
    efficiency_data = efficiency_data.sort_values('efficiency', ascending=False)
    
    # Create color scale based on efficiency values
    fig = px.bar(
        efficiency_data, 
        x='location', 
        y='efficiency',
        title='Signal Efficiency by Location',
        labels={'location': 'Location', 'efficiency': 'Efficiency (%)'},
        color='efficiency',
        color_continuous_scale=['red', 'yellow', 'green'],
        range_color=[0, 100]
    )
    
    return fig

def filter_incidents(df, incident_type, severity):
    """
    Filter incidents by type and severity
    
    Args:
        df: Incidents dataframe
        incident_type: Selected incident type (or 'All')
        severity: Selected severity level (or 'All')
        
    Returns:
        Filtered dataframe
    """
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply type filter
    if incident_type != 'All':
        filtered_df = filtered_df[filtered_df['type'] == incident_type]
    
    # Apply severity filter
    if severity != 'All':
        filtered_df = filtered_df[filtered_df['severity'] == severity]
    
    return filtered_df

def plot_incidents_by_type(df):
    """
    Create a pie chart of incidents by type
    
    Args:
        df: Incidents dataframe
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return px.pie()
    
    # Count incidents by type
    type_counts = df['type'].value_counts().reset_index()
    type_counts.columns = ['type', 'count']
    
    # Create pie chart
    fig = px.pie(
        type_counts, 
        values='count', 
        names='type',
        title='Incidents by Type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return fig

def plot_incidents_by_severity(df):
    """
    Create a pie chart of incidents by severity
    
    Args:
        df: Incidents dataframe
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return px.pie()
    
    # Count incidents by severity
    severity_counts = df['severity'].value_counts().reset_index()
    severity_counts.columns = ['severity', 'count']
    
    # Create color map for severity levels
    color_map = {
        'Low': 'green',
        'Medium': 'yellow',
        'High': 'orange',
        'Critical': 'red'
    }
    
    # Create pie chart
    fig = px.pie(
        severity_counts, 
        values='count', 
        names='severity',
        title='Incidents by Severity',
        color='severity',
        color_discrete_map=color_map
    )
    
    return fig

def plot_traffic_by_day_of_week(df):
    """
    Create a bar chart of traffic volume by day of week
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return px.bar()
    
    # Extract day of week
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
    
    # Group by day of week and calculate mean volume
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_data = df.groupby('day_of_week').agg({'volume': 'mean'}).reset_index()
    
    # Ensure correct day order
    dow_data['day_of_week'] = pd.Categorical(dow_data['day_of_week'], categories=day_order, ordered=True)
    dow_data = dow_data.sort_values('day_of_week')
    
    # Create bar chart
    fig = px.bar(
        dow_data, 
        x='day_of_week', 
        y='volume',
        title='Average Traffic Volume by Day of Week',
        labels={'day_of_week': 'Day of Week', 'volume': 'Average Traffic Volume'}
    )
    
    return fig

def plot_traffic_by_time_of_day(df):
    """
    Create a line chart of traffic volume by time of day
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return px.line()
    
    # Extract hour of day
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    
    # Group by hour of day and calculate mean volume
    tod_data = df.groupby('hour_of_day').agg({'volume': 'mean'}).reset_index()
    
    # Create line chart
    fig = px.line(
        tod_data, 
        x='hour_of_day', 
        y='volume',
        title='Average Traffic Volume by Time of Day',
        labels={'hour_of_day': 'Hour of Day (24h)', 'volume': 'Average Traffic Volume'}
    )
    
    # Update x-axis to show all hours
    fig.update_xaxes(tickvals=list(range(0, 24)))
    
    return fig

def plot_traffic_volume_trend(df):
    """
    Create a line chart of traffic volume trend over time
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return px.line()
    
    # Group by day and calculate mean volume
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_data = df.groupby('date').agg({'volume': 'mean'}).reset_index()
    
    # Create line chart with trend line
    fig = px.scatter(
        daily_data, 
        x='date', 
        y='volume',
        title='Traffic Volume Trend',
        labels={'date': 'Date', 'volume': 'Average Traffic Volume'},
        trendline='ols'
    )
    
    return fig

def plot_incident_correlation(traffic_df, incidents_df):
    """
    Create a scatter plot showing correlation between traffic volume and incident count
    
    Args:
        traffic_df: Traffic data dataframe
        incidents_df: Incidents dataframe
        
    Returns:
        Plotly figure object
    """
    if traffic_df.empty or incidents_df.empty:
        return px.scatter()
    
    # Group traffic data by day
    traffic_df['date'] = pd.to_datetime(traffic_df['timestamp']).dt.date
    daily_traffic = traffic_df.groupby('date').agg({'volume': 'mean'}).reset_index()
    
    # Group incidents by day
    incidents_df['date'] = pd.to_datetime(incidents_df['timestamp']).dt.date
    daily_incidents = incidents_df.groupby('date').size().reset_index(name='incident_count')
    
    # Merge the data
    merged_data = pd.merge(daily_traffic, daily_incidents, on='date', how='left')
    merged_data['incident_count'] = merged_data['incident_count'].fillna(0)
    
    # Create scatter plot
    fig = px.scatter(
        merged_data, 
        x='volume', 
        y='incident_count',
        title='Correlation: Traffic Volume vs. Incident Count',
        labels={'volume': 'Average Traffic Volume', 'incident_count': 'Number of Incidents'},
        trendline='ols'
    )
    
    return fig

def calculate_incident_correlation(traffic_df, incidents_df):
    """
    Calculate correlation coefficient between traffic volume and incident count
    
    Args:
        traffic_df: Traffic data dataframe
        incidents_df: Incidents dataframe
        
    Returns:
        Correlation coefficient
    """
    if traffic_df.empty or incidents_df.empty:
        return 0
    
    # Group traffic data by day
    traffic_df['date'] = pd.to_datetime(traffic_df['timestamp']).dt.date
    daily_traffic = traffic_df.groupby('date').agg({'volume': 'mean'}).reset_index()
    
    # Group incidents by day
    incidents_df['date'] = pd.to_datetime(incidents_df['timestamp']).dt.date
    daily_incidents = incidents_df.groupby('date').size().reset_index(name='incident_count')
    
    # Merge the data
    merged_data = pd.merge(daily_traffic, daily_incidents, on='date', how='left')
    merged_data['incident_count'] = merged_data['incident_count'].fillna(0)
    
    # Calculate correlation
    correlation = merged_data['volume'].corr(merged_data['incident_count'])
    
    return correlation

def get_busiest_day(df):
    """
    Get the busiest and least busy day of the week
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Tuple of (busiest_day, least_busy_day)
    """
    if df.empty:
        return ('N/A', 'N/A')
    
    # Extract day of week
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
    
    # Group by day of week and calculate mean volume
    dow_data = df.groupby('day_of_week').agg({'volume': 'mean'})
    
    # Get busiest and least busy days
    busiest_day = dow_data['volume'].idxmax()
    least_busy_day = dow_data['volume'].idxmin()
    
    return (busiest_day, least_busy_day)

def get_peak_hours(df):
    """
    Get the morning and evening peak hours
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Tuple of (morning_peak, evening_peak)
    """
    if df.empty:
        return ('N/A', 'N/A')
    
    # Extract hour of day
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    
    # Group by hour of day and calculate mean volume
    hour_data = df.groupby('hour_of_day').agg({'volume': 'mean'})
    
    # Find morning peak (5-12)
    morning_data = hour_data.loc[5:12]
    morning_peak = morning_data['volume'].idxmax()
    
    # Find evening peak (13-20)
    evening_data = hour_data.loc[13:20]
    evening_peak = evening_data['volume'].idxmax()
    
    return (f"{morning_peak}:00", f"{evening_peak}:00")

def describe_trend(df):
    """
    Generate a description of the traffic volume trend
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Description of trend
    """
    if df.empty:
        return "No data available for trend analysis."
    
    # Group by day and calculate mean volume
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_data = df.groupby('date').agg({'volume': 'mean'}).reset_index()
    
    # Calculate simple linear regression
    x = np.arange(len(daily_data))
    y = daily_data['volume'].values
    
    if len(x) < 2:
        return "Insufficient data for trend analysis."
    
    slope, _ = np.polyfit(x, y, 1)
    
    # Interpret the trend
    if slope > 5:
        trend = "strongly increasing"
    elif slope > 1:
        trend = "moderately increasing"
    elif slope > 0:
        trend = "slightly increasing"
    elif slope == 0:
        trend = "stable"
    elif slope > -1:
        trend = "slightly decreasing"
    elif slope > -5:
        trend = "moderately decreasing"
    else:
        trend = "strongly decreasing"
    
    # Calculate percentage change
    start_avg = daily_data['volume'].iloc[:min(7, len(daily_data))].mean()
    end_avg = daily_data['volume'].iloc[-min(7, len(daily_data)):].mean()
    
    if start_avg > 0:
        percent_change = ((end_avg - start_avg) / start_avg) * 100
        direction = "increase" if percent_change > 0 else "decrease"
        
        return f"Traffic volume shows a {trend} trend. There has been a {abs(percent_change):.1f}% {direction} in average volume over the analyzed period."
    else:
        return f"Traffic volume shows a {trend} trend."

def plot_traffic_flow(df):
    """
    Create a line chart showing traffic flow patterns
    
    Args:
        df: Traffic data dataframe
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return px.line()
    
    # Extract hour of day and day of week
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
    
    # Group by hour and day, calculate mean volume
    grouped_data = df.groupby(['day_of_week', 'hour_of_day']).agg({'volume': 'mean'}).reset_index()
    
    # Ensure correct day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grouped_data['day_of_week'] = pd.Categorical(grouped_data['day_of_week'], categories=day_order, ordered=True)
    grouped_data = grouped_data.sort_values(['day_of_week', 'hour_of_day'])
    
    # Create line chart
    fig = px.line(
        grouped_data, 
        x='hour_of_day', 
        y='volume',
        color='day_of_week',
        title='Traffic Flow Patterns by Day and Hour',
        labels={'hour_of_day': 'Hour of Day (24h)', 'volume': 'Average Traffic Volume', 'day_of_week': 'Day of Week'}
    )
    
    # Update x-axis to show all hours
    fig.update_xaxes(tickvals=list(range(0, 24)))
    
    return fig

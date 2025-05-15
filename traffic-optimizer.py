import pandas as pd
import numpy as np
import datetime

def optimize_signal_timing(traffic_df, current_cycle_length):
    """
    Calculate optimized signal timing based on traffic patterns
    
    Args:
        traffic_df: Traffic data dataframe for a specific location
        current_cycle_length: Current signal cycle length in seconds
        
    Returns:
        Optimized cycle length in seconds
    """
    if traffic_df.empty:
        return current_cycle_length
    
    # Calculate average volume and congestion
    avg_volume = traffic_df['volume'].mean()
    avg_congestion = traffic_df['congestion'].mean()
    
    # Simple optimization logic based on volume and congestion
    # In a real system, this would use more sophisticated algorithms
    
    # Base cycle length adjustment
    if avg_volume < 200:
        # Low volume, shorter cycles
        base_cycle = 60
    elif avg_volume < 400:
        # Medium volume
        base_cycle = 90
    else:
        # High volume, longer cycles
        base_cycle = 120
    
    # Congestion adjustment
    if avg_congestion < 0.3:
        # Low congestion, shorter cycles
        congestion_factor = 0.8
    elif avg_congestion < 0.7:
        # Medium congestion
        congestion_factor = 1.0
    else:
        # High congestion, longer cycles
        congestion_factor = 1.2
    
    # Calculate optimized cycle length
    optimized_cycle = int(base_cycle * congestion_factor)
    
    # Ensure cycle length is within reasonable bounds
    optimized_cycle = max(45, min(optimized_cycle, 180))
    
    return optimized_cycle

def calculate_optimization_impact(traffic_df, current_cycle, optimized_cycle):
    """
    Calculate the estimated impact of signal timing optimization
    
    Args:
        traffic_df: Traffic data dataframe
        current_cycle: Current signal cycle length in seconds
        optimized_cycle: Optimized signal cycle length in seconds
        
    Returns:
        Estimated percentage improvement in traffic flow
    """
    if traffic_df.empty or current_cycle == optimized_cycle:
        return 0.0
    
    # Calculate average volume and congestion
    avg_congestion = traffic_df['congestion'].mean()
    
    # Estimate current delay
    current_delay = estimate_delay(avg_congestion, current_cycle)
    
    # Estimate optimized delay
    optimized_delay = estimate_delay(avg_congestion, optimized_cycle)
    
    # Calculate improvement
    if current_delay > 0:
        improvement = ((current_delay - optimized_delay) / current_delay) * 100
        return improvement
    else:
        return 0.0

def estimate_delay(congestion, cycle_length):
    """
    Estimate delay based on congestion and cycle length
    
    Args:
        congestion: Congestion level (0-1)
        cycle_length: Signal cycle length in seconds
        
    Returns:
        Estimated delay in seconds
    """
    # Simple delay model
    # In a real system, this would use more sophisticated models
    base_delay = cycle_length * 0.5  # Average wait is half the cycle length
    congestion_factor = 1 + (congestion * 2)  # Congestion increases delay
    
    # Calculate delay
    delay = base_delay * congestion_factor
    
    return delay

def suggest_alternative_routes(location, traffic_df, incidents_df):
    """
    Suggest alternative routes based on current traffic and incidents
    
    Args:
        location: Current location
        traffic_df: Traffic data dataframe
        incidents_df: Incidents dataframe
        
    Returns:
        List of alternative route dictionaries
    """
    if traffic_df.empty:
        return []
    
    # Get all locations
    all_locations = traffic_df['location'].unique()
    
    # Filter out current location
    other_locations = [loc for loc in all_locations if loc != location]
    
    if not other_locations:
        return []
    
    # Get recent traffic data
    recent_timestamp = traffic_df['timestamp'].max()
    start_time = pd.to_datetime(recent_timestamp) - pd.Timedelta(hours=1)
    recent_traffic = traffic_df[pd.to_datetime(traffic_df['timestamp']) >= start_time]
    
    # Check for active incidents
    recent_incidents = incidents_df[incidents_df['status'] == 'Active']
    
    # Locations with active incidents
    incident_locations = set(recent_incidents['location'])
    
    # Calculate congestion for each location
    location_congestion = {}
    for loc in all_locations:
        loc_data = recent_traffic[recent_traffic['location'] == loc]
        if not loc_data.empty:
            location_congestion[loc] = loc_data['congestion'].mean()
    
    # Find potential alternative routes
    alternatives = []
    
    for alt_loc in other_locations:
        # Skip locations with active incidents
        if alt_loc in incident_locations:
            continue
        
        # Check if alternative has lower congestion
        if alt_loc in location_congestion and location in location_congestion:
            current_congestion = location_congestion[location]
            alt_congestion = location_congestion[alt_loc]
            
            if alt_congestion < current_congestion:
                # Calculate time saving (rough estimate)
                time_saving = (current_congestion - alt_congestion) * 15  # 15 minutes maximum saving
                
                # Create random distance for demo (would be calculated using actual distances in real system)
                distance = round(np.random.uniform(1.5, 5.0), 1)
                
                alternatives.append({
                    'description': f"Route via {alt_loc}",
                    'congestion': alt_congestion,
                    'time_saving': round(time_saving, 1),
                    'distance': distance
                })
    
    # Sort by time saving (descending)
    alternatives.sort(key=lambda x: x['time_saving'], reverse=True)
    
    # Return top 3 alternatives
    return alternatives[:3]

def generate_recommendations(traffic_df, signal_df, incidents_df):
    """
    Generate traffic optimization recommendations
    
    Args:
        traffic_df: Traffic data dataframe
        signal_df: Traffic signals dataframe
        incidents_df: Incidents dataframe
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if traffic_df.empty:
        return ["Insufficient traffic data for optimization recommendations."]
    
    # Extract traffic patterns
    traffic_df['hour'] = pd.to_datetime(traffic_df['timestamp']).dt.hour
    traffic_df['day_of_week'] = pd.to_datetime(traffic_df['timestamp']).dt.day_name()
    
    # Find peak hours
    hourly_data = traffic_df.groupby('hour').agg({'volume': 'mean'}).reset_index()
    morning_peak = hourly_data.loc[(hourly_data['hour'] >= 6) & (hourly_data['hour'] <= 10), 'volume'].idxmax()
    morning_peak_hour = hourly_data.loc[morning_peak, 'hour'] if morning_peak is not None else None
    
    evening_peak = hourly_data.loc[(hourly_data['hour'] >= 15) & (hourly_data['hour'] <= 19), 'volume'].idxmax()
    evening_peak_hour = hourly_data.loc[evening_peak, 'hour'] if evening_peak is not None else None
    
    # Check signal efficiency
    if not signal_df.empty:
        avg_efficiency = signal_df['efficiency'].mean()
        low_efficiency_signals = signal_df[signal_df['efficiency'] < 75]
        
        if not low_efficiency_signals.empty:
            for _, signal in low_efficiency_signals.iterrows():
                recommendations.append(
                    f"Improve signal efficiency at {signal['location']} (current efficiency: {signal['efficiency']}%)."
                )
        
        if avg_efficiency < 85:
            recommendations.append(
                "Consider system-wide signal timing optimization to improve overall efficiency."
            )
    
    # Recommendations based on peak hours
    if morning_peak_hour is not None:
        recommendations.append(
            f"Adjust signal timing during morning peak hours ({int(morning_peak_hour)}:00) to prioritize main commuting routes."
        )
    
    if evening_peak_hour is not None:
        recommendations.append(
            f"Extend green light durations during evening peak hours ({int(evening_peak_hour)}:00) on major arterial roads."
        )
    
    # Check for recurring congestion patterns
    location_congestion = traffic_df.groupby('location').agg({'congestion': 'mean'}).reset_index()
    high_congestion_locations = location_congestion[location_congestion['congestion'] > 0.7]
    
    if not high_congestion_locations.empty:
        for _, location in high_congestion_locations.iterrows():
            recommendations.append(
                f"Address recurring congestion at {location['location']} through improved signal coordination or road capacity improvements."
            )
    
    # Check incidents impact
    if not incidents_df.empty:
        recent_incidents = incidents_df[incidents_df['status'] == 'Active']
        high_impact_incidents = recent_incidents[recent_incidents['impact'] > 0.5]
        
        if not high_impact_incidents.empty:
            for _, incident in high_impact_incidents.iterrows():
                recommendations.append(
                    f"Implement temporary traffic management plan around {incident['location']} due to {incident['type'].lower()} incident."
                )
    
    # General recommendations
    recommendations.append(
        "Consider implementing adaptive signal control technology to automatically adjust to changing traffic conditions."
    )
    
    if not recommendations:
        recommendations.append("No specific optimization recommendations at this time.")
    
    return recommendations

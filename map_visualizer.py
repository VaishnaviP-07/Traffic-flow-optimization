import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np

def create_traffic_map(traffic_df, incidents_df=None):
    """
    Create an interactive map visualizing traffic volume and incidents
    
    Args:
        traffic_df: Traffic data dataframe
        incidents_df: Incidents data dataframe (optional)
        
    Returns:
        Folium map object
    """
    # Check if dataframe is empty
    if traffic_df.empty:
        # Create a default map
        default_center = [40.7128, -74.0060]  # NYC coordinates as default
        traffic_map = folium.Map(location=default_center, zoom_start=12)
        folium.Marker(
            location=default_center,
            popup="No traffic data available",
            icon=folium.Icon(icon="info-sign", color="red")
        ).add_to(traffic_map)
        return traffic_map
    
    # Calculate map center based on data
    center_lat = traffic_df['latitude'].mean()
    center_lon = traffic_df['longitude'].mean()
    
    # Create base map
    traffic_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add traffic volume heatmap
    if len(traffic_df) > 0:
        # Aggregate data by location
        location_data = traffic_df.groupby(['latitude', 'longitude']).agg({'volume': 'mean'}).reset_index()
        
        # Create heat map data
        heat_data = [[row['latitude'], row['longitude'], row['volume']] for _, row in location_data.iterrows()]
        
        # Add heat map layer
        HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(traffic_map)
    
    # Add traffic markers with popups
    location_groups = traffic_df.groupby('location')
    marker_cluster = MarkerCluster().add_to(traffic_map)
    
    for location, group in location_groups:
        lat = group['latitude'].mean()
        lon = group['longitude'].mean()
        volume = group['volume'].mean()
        speed = group['speed'].mean()
        congestion = group['congestion'].mean()
        
        # Color based on congestion level
        if congestion < 0.3:
            color = "green"
        elif congestion < 0.7:
            color = "orange"
        else:
            color = "red"
        
        # Create popup content
        popup_content = f"""
        <b>{location}</b><br>
        Average Volume: {int(volume)}<br>
        Average Speed: {speed:.1f} mph<br>
        Congestion Level: {congestion:.2f}
        """
        
        # Add marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=200),
            icon=folium.Icon(icon="car", prefix="fa", color=color)
        ).add_to(marker_cluster)
    
    # Add incidents if provided
    if incidents_df is not None and not incidents_df.empty:
        # Add active incidents to the map
        active_incidents = incidents_df[incidents_df['status'] == 'Active']
        
        for _, incident in active_incidents.iterrows():
            # Choose icon and color based on incident type and severity
            icon_map = {
                "Accident": "car-crash",
                "Construction": "hard-hat",
                "Event": "calendar",
                "Hazard": "exclamation-triangle",
                "Other": "info-circle"
            }
            
            color_map = {
                "Low": "green",
                "Medium": "orange",
                "High": "red",
                "Critical": "darkred"
            }
            
            icon_type = icon_map.get(incident['type'], "info-circle")
            color = color_map.get(incident['severity'], "blue")
            
            # Create popup content
            popup_content = f"""
            <b>{incident['type']} ({incident['severity']})</b><br>
            Location: {incident['location']}<br>
            Description: {incident['description']}<br>
            Impact: {incident['impact']:.2f}<br>
            Status: {incident['status']}
            """
            
            # Add marker
            folium.Marker(
                location=[incident['latitude'], incident['longitude']],
                popup=folium.Popup(popup_content, max_width=200),
                icon=folium.Icon(icon=icon_type, prefix="fa", color=color)
            ).add_to(traffic_map)
    
    return traffic_map

def create_signal_map(signals_df):
    """
    Create an interactive map visualizing traffic signals
    
    Args:
        signals_df: Traffic signals dataframe
        
    Returns:
        Folium map object
    """
    # Check if dataframe is empty
    if signals_df.empty:
        # Create a default map
        default_center = [40.7128, -74.0060]  # NYC coordinates as default
        signal_map = folium.Map(location=default_center, zoom_start=12)
        folium.Marker(
            location=default_center,
            popup="No signal data available",
            icon=folium.Icon(icon="info-sign", color="red")
        ).add_to(signal_map)
        return signal_map
    
    # Calculate map center based on data
    center_lat = signals_df['latitude'].mean()
    center_lon = signals_df['longitude'].mean()
    
    # Create base map
    signal_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add signal markers
    for _, signal in signals_df.iterrows():
        # Choose color based on status and efficiency
        if signal['status'] == 'Issue':
            color = "red"
        elif signal['efficiency'] < 80:
            color = "orange"
        else:
            color = "green"
        
        # Create popup content
        popup_content = f"""
        <b>Signal ID: {signal['signal_id']}</b><br>
        Location: {signal['location']}<br>
        Type: {signal['type']}<br>
        Status: {signal['status']}<br>
        Efficiency: {signal['efficiency']}%<br>
        Cycle Length: {signal['cycle_length']} sec<br>
        Last Maintenance: {signal['last_maintenance']}
        """
        
        # Add marker
        folium.Marker(
            location=[signal['latitude'], signal['longitude']],
            popup=folium.Popup(popup_content, max_width=200),
            icon=folium.Icon(icon="traffic-light", prefix="fa", color=color)
        ).add_to(signal_map)
    
    return signal_map

def create_incident_map(incidents_df):
    """
    Create an interactive map visualizing traffic incidents
    
    Args:
        incidents_df: Incidents dataframe
        
    Returns:
        Folium map object
    """
    # Check if dataframe is empty
    if incidents_df.empty:
        # Create a default map
        default_center = [40.7128, -74.0060]  # NYC coordinates as default
        incident_map = folium.Map(location=default_center, zoom_start=12)
        folium.Marker(
            location=default_center,
            popup="No incident data available",
            icon=folium.Icon(icon="info-sign", color="red")
        ).add_to(incident_map)
        return incident_map
    
    # Calculate map center based on data
    center_lat = incidents_df['latitude'].mean()
    center_lon = incidents_df['longitude'].mean()
    
    # Create base map
    incident_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Create marker clusters
    marker_cluster = MarkerCluster().add_to(incident_map)
    
    # Add incident markers
    for _, incident in incidents_df.iterrows():
        # Choose icon and color based on incident type and severity
        icon_map = {
            "Accident": "car-crash",
            "Construction": "hard-hat",
            "Event": "calendar",
            "Hazard": "exclamation-triangle",
            "Other": "info-circle"
        }
        
        color_map = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
            "Critical": "darkred"
        }
        
        icon_type = icon_map.get(incident['type'], "info-circle")
        color = color_map.get(incident['severity'], "blue")
        
        # Create popup content
        popup_content = f"""
        <b>{incident['type']} ({incident['severity']})</b><br>
        Location: {incident['location']}<br>
        Description: {incident['description']}<br>
        Time: {incident['timestamp']}<br>
        Duration: {incident['duration']} hours<br>
        Impact: {incident['impact']:.2f}<br>
        Status: {incident['status']}
        """
        
        # Add marker
        folium.Marker(
            location=[incident['latitude'], incident['longitude']],
            popup=folium.Popup(popup_content, max_width=200),
            icon=folium.Icon(icon=icon_type, prefix="fa", color=color)
        ).add_to(marker_cluster)
    
    return incident_map

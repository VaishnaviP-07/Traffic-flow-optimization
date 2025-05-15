import pandas as pd
import streamlit as st
import os
import datetime
import numpy as np

def load_traffic_data():
    """
    Load traffic data from CSV file or create sample data if file doesn't exist
    """
    try:
        # Try to load from file
        if os.path.exists("sample_data/traffic_data.csv"):
            data = pd.read_csv("sample_data/traffic_data.csv")
            return data
        else:
            # Create sample data for demonstration
            return create_sample_traffic_data()
    except Exception as e:
        st.error(f"Error loading traffic data: {str(e)}")
        return pd.DataFrame()

def load_incidents_data():
    """
    Load incidents data from CSV file or create sample data if file doesn't exist
    """
    try:
        # Try to load from file
        if os.path.exists("sample_data/incidents.csv"):
            data = pd.read_csv("sample_data/incidents.csv")
            return data
        else:
            # Create sample data for demonstration
            return create_sample_incidents_data()
    except Exception as e:
        st.error(f"Error loading incidents data: {str(e)}")
        return pd.DataFrame()

def load_signals_data():
    """
    Load traffic signals data from CSV file or create sample data if file doesn't exist
    """
    try:
        # Try to load from file
        if os.path.exists("sample_data/traffic_signals.csv"):
            data = pd.read_csv("sample_data/traffic_signals.csv")
            return data
        else:
            # Create sample data for demonstration
            return create_sample_signals_data()
    except Exception as e:
        st.error(f"Error loading traffic signals data: {str(e)}")
        return pd.DataFrame()

def create_sample_traffic_data():
    """
    Create sample traffic data for demonstration purposes
    """
    # Sample locations
    locations = [
        "Main St & 1st Ave", "Broadway & 5th St", "Park Rd & Oak Ln",
        "Highway 101 & Exit 25", "Central Ave & Washington St"
    ]
    
    # Generate dates for the past 90 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=90)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create dataframe structure
    data = []
    
    # Generate sample data for each location and date
    for location in locations:
        for timestamp in date_range:
            # Base traffic volume (varies by location)
            base_volume = np.random.randint(100, 500)
            
            # Add daily pattern (more traffic during morning and evening rush hours)
            hour = timestamp.hour
            if 7 <= hour <= 9:  # Morning rush
                volume_factor = np.random.uniform(1.5, 2.0)
            elif 16 <= hour <= 18:  # Evening rush
                volume_factor = np.random.uniform(1.6, 2.2)
            elif 0 <= hour <= 5:  # Late night
                volume_factor = np.random.uniform(0.1, 0.3)
            else:  # Regular hours
                volume_factor = np.random.uniform(0.7, 1.3)
            
            # Add weekly pattern (less traffic on weekends)
            if timestamp.dayofweek >= 5:  # Weekend
                volume_factor *= np.random.uniform(0.5, 0.7)
            
            volume = int(base_volume * volume_factor)
            speed = np.random.randint(5, 60)  # Speed in mph
            congestion = np.random.uniform(0, 1)  # Congestion level (0-1)
            
            # Calculate travel time (minutes) based on congestion
            travel_time = 5 + 20 * congestion
            
            data.append({
                'timestamp': timestamp,
                'location': location,
                'volume': volume,
                'speed': speed,
                'congestion': congestion,
                'travel_time': travel_time,
                'latitude': np.random.uniform(40.7, 40.8),  # Random coordinates for demo
                'longitude': np.random.uniform(-74.0, -73.9)
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV for future use
    os.makedirs("sample_data", exist_ok=True)
    df.to_csv("sample_data/traffic_data.csv", index=False)
    
    return df

def create_sample_incidents_data():
    """
    Create sample traffic incident data for demonstration purposes
    """
    # Sample incident types
    incident_types = ["Accident", "Construction", "Event", "Hazard", "Other"]
    severity_levels = ["Low", "Medium", "High", "Critical"]
    
    # Sample locations (same as traffic data for consistency)
    locations = [
        "Main St & 1st Ave", "Broadway & 5th St", "Park Rd & Oak Ln",
        "Highway 101 & Exit 25", "Central Ave & Washington St"
    ]
    
    # Generate dates for the past 90 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=90)
    
    # Create random incidents
    data = []
    num_incidents = 150  # Number of sample incidents
    
    for _ in range(num_incidents):
        # Random timestamp within the date range
        days_offset = np.random.randint(0, 90)
        hours_offset = np.random.randint(0, 24)
        timestamp = end_date - datetime.timedelta(days=days_offset, hours=hours_offset)
        
        # Random duration in hours
        duration = np.random.randint(1, 8)
        
        # Random location, type, and severity
        location = np.random.choice(locations)
        incident_type = np.random.choice(incident_types)
        severity = np.random.choice(severity_levels)
        
        # Impact on traffic (percentage increase in congestion)
        impact = np.random.uniform(0.1, 0.9)
        
        # Description of incident
        descriptions = {
            "Accident": ["Vehicle collision", "Multi-car accident", "Fender bender"],
            "Construction": ["Road repair", "Lane closure", "Utility work"],
            "Event": ["Parade", "Sports event", "Concert traffic"],
            "Hazard": ["Debris on road", "Flooding", "Fallen tree"],
            "Other": ["Stalled vehicle", "Police activity", "Traffic signal malfunction"]
        }
        description = np.random.choice(descriptions[incident_type])
        
        # Generate coordinates near the city center
        latitude = np.random.uniform(40.7, 40.8)
        longitude = np.random.uniform(-74.0, -73.9)
        
        data.append({
            'timestamp': timestamp,
            'location': location,
            'type': incident_type,
            'severity': severity,
            'duration': duration,
            'impact': impact,
            'description': description,
            'latitude': latitude,
            'longitude': longitude,
            'status': "Resolved" if timestamp < end_date - datetime.timedelta(hours=duration) else "Active"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV for future use
    os.makedirs("sample_data", exist_ok=True)
    df.to_csv("sample_data/incidents.csv", index=False)
    
    return df

def create_sample_signals_data():
    """
    Create sample traffic signal data for demonstration purposes
    """
    # Sample locations (same as traffic data for consistency)
    locations = [
        "Main St & 1st Ave", "Broadway & 5th St", "Park Rd & Oak Ln",
        "Highway 101 & Exit 25", "Central Ave & Washington St"
    ]
    
    # Create sample traffic signal data
    data = []
    
    for location in locations:
        # Create 1-3 signals per location
        num_signals = np.random.randint(1, 4)
        
        for i in range(num_signals):
            signal_id = f"SIG-{locations.index(location)}-{i}"
            
            # Random signal timing parameters
            cycle_length = np.random.choice([60, 90, 120, 150])  # Cycle length in seconds
            green_time = int(cycle_length * np.random.uniform(0.4, 0.7))  # Green time in seconds
            
            # Status (mostly operational with a few issues)
            status = np.random.choice(["Operational", "Issue"], p=[0.9, 0.1])
            
            # Random installation date
            install_year = np.random.randint(2010, 2022)
            install_month = np.random.randint(1, 13)
            install_day = np.random.randint(1, 29)
            install_date = f"{install_year}-{install_month:02d}-{install_day:02d}"
            
            # Random last maintenance date (within the past year)
            days_since_maintenance = np.random.randint(0, 365)
            maintenance_date = (datetime.datetime.now() - datetime.timedelta(days=days_since_maintenance)).strftime("%Y-%m-%d")
            
            # Efficiency score (0-100)
            efficiency = np.random.randint(70, 100) if status == "Operational" else np.random.randint(30, 70)
            
            # Coordinates
            latitude = np.random.uniform(40.7, 40.8)
            longitude = np.random.uniform(-74.0, -73.9)
            
            data.append({
                'signal_id': signal_id,
                'location': location,
                'status': status,
                'cycle_length': cycle_length,
                'green_time': green_time,
                'installation_date': install_date,
                'last_maintenance': maintenance_date,
                'efficiency': efficiency,
                'type': np.random.choice(["Fixed-time", "Actuated", "Adaptive"]),
                'latitude': latitude,
                'longitude': longitude
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV for future use
    os.makedirs("sample_data", exist_ok=True)
    df.to_csv("sample_data/traffic_signals.csv", index=False)
    
    return df

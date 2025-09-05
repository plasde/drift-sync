import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt

import requests
import json
import geopandas as gpd
from shapely.geometry import LineString, Point

def get_marine_features_direct(box):
    """Get coastline and nautical features directly from Overpass API"""
    north, south, east, west = box
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Query for coastline
    coastline_query = f"""
    [out:json][timeout:30];
    (
      way["natural"="coastline"](bbox:{south},{west},{north},{east});
      relation["natural"="coastline"](bbox:{south},{west},{north},{east});
    );
    out geom;
    """
    
    # Query for nautical features
    nautical_query = f"""
    [out:json][timeout:30];
    (
      node["seamark:type"](bbox:{south},{west},{north},{east});
      way["seamark:type"](bbox:{south},{west},{north},{east});
      node["man_made"="lighthouse"](bbox:{south},{west},{north},{east});
      node["seamark:buoy_lateral:category"](bbox:{south},{west},{north},{east});
    );
    out geom;
    """
    
    def query_overpass(query):
        try:
            response = requests.get(overpass_url, params={'data': query}, timeout=30)
            return response.json()
        except Exception as e:
            print(f"Query error: {e}")
            return {'elements': []}
    
    # Get coastline data
    coastline_data = query_overpass(coastline_query)
    coastline_geoms = []
    for element in coastline_data.get('elements', []):
        if element['type'] == 'way' and 'geometry' in element:
            coords = [(node['lon'], node['lat']) for node in element['geometry']]
            if len(coords) > 1:
                coastline_geoms.append(LineString(coords))
    
    # Get nautical data
    nautical_data = query_overpass(nautical_query)
    nautical_geoms = []
    nautical_info = []
    for element in nautical_data.get('elements', []):
        if element['type'] == 'node':
            point = Point(element['lon'], element['lat'])
            nautical_geoms.append(point)
            # Get feature info
            tags = element.get('tags', {})
            feature_type = tags.get('seamark:type', tags.get('man_made', 'unknown'))
            nautical_info.append(feature_type)
        elif element['type'] == 'way' and 'geometry' in element:
            coords = [(node['lon'], node['lat']) for node in element['geometry']]
            if len(coords) > 1:
                nautical_geoms.append(LineString(coords))
                tags = element.get('tags', {})
                feature_type = tags.get('seamark:type', 'unknown')
                nautical_info.append(feature_type)
    
    # Create GeoDataFrames
    coastline_gdf = gpd.GeoDataFrame({'geometry': coastline_geoms}) if coastline_geoms else gpd.GeoDataFrame()
    nautical_gdf = gpd.GeoDataFrame({
        'geometry': nautical_geoms, 
        'type': nautical_info
    }) if nautical_geoms else gpd.GeoDataFrame()
    
    return coastline_gdf, nautical_gdf

def plot_marine_features(box, title="Marine Features"):
    """Plot coastline and nautical features for a given bounding box"""
    print(f"Querying area: {box}")
    coastline, nautical = get_marine_features_direct(box)
    
    print(f"Found {len(coastline)} coastline features")
    print(f"Found {len(nautical)} nautical features")
    
    if not nautical.empty:
        print("Nautical feature types:", nautical['type'].value_counts().to_dict())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot coastline
    if not coastline.empty:
        coastline.plot(ax=ax, color='blue', linewidth=2, label=f'Coastline ({len(coastline)})')
    
    # Plot nautical features
    if not nautical.empty:
        # Different colors for different types
        colors = {'lighthouse': 'red', 'buoy_lateral': 'orange', 'buoy_cardinal': 'yellow', 
                 'beacon': 'purple', 'light': 'pink'}
        
        for feature_type in nautical['type'].unique():
            subset = nautical[nautical['type'] == feature_type]
            color = colors.get(feature_type, 'red')
            subset.plot(ax=ax, color=color, markersize=50, label=f'{feature_type} ({len(subset)})')
    
    # Set extent and labels
    north, south, east, west = box
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    
    ax.set_title(f"{title}\nBox: {box}", fontsize=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return coastline, nautical

# Test with your Dover area
dover_box = (51.2, 51.1, 1.4, 1.3)
coastline, nautical = plot_marine_features(dover_box, "Dover Area Marine Features")

# Now you can test random areas easily:
def test_random_coastal_areas():
    """Test several small coastal areas"""
    test_areas = [
        ((51.13, 51.12, 1.32, 1.31), "Dover"),
        ((50.77, 50.76, -1.11, -1.12), "Portsmouth"), 
        ((53.42, 53.41, -3.02, -3.03), "Liverpool"),
        ((55.95, 55.94, -3.20, -3.21), "Edinburgh"),
    ]
    
    for box, name in test_areas:
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        coastline, nautical = plot_marine_features(box, f"{name} Marine Features")
        
        # Brief pause between plots
        input("Press Enter for next area...")

# Uncomment to test multiple areas:

#test_random_coastal_areas()
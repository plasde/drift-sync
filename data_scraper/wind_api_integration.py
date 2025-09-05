import requests
import numpy as np
from math import radians, degrees, sin, cos
import time
from typing import Tuple, Optional, Dict, Any


class WeatherAPIWindField:
    """
    Fetches real wind data from OpenWeatherMap API and converts it
    to work with your existing WindField class structure
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather API client

        Args:
            api_key: OpenWeatherMap API key. Get free key from: https://openweathermap.org/api
                    If None, will use dummy data for testing
        """
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"

    def create_wind_field_from_api(
        self,
        bounds: Tuple[int, int, int, int],
        bbox: Tuple[float, float, float, float],
        resolution: float = 0.1,
    ) -> "WindField":
        """
        Create a WindField object populated with real API data

        Args:
            width: Grid width for your existing WindField
            height: Grid height for your existing WindField
            bbox: (north, south, east, west) bounding box in degrees
            resolution: Grid resolution in degrees

        Returns:
            WindField object compatible with your a_star pathfinder
        """
        from core.wind import WindField  # Import your existing WindField class

        # Create the wind field object
        wind_field = WindField(bounds=bounds, resolution=resolution)

        # Get real wind data from API
        wind_data = self._fetch_wind_grid(bbox, resolution)

        # Convert API data to your WindField format
        self._populate_wind_field(wind_field, wind_data, bbox)

        return wind_field

    def _fetch_wind_grid(
        self, bbox: Tuple[float, float, float, float], resolution: float
    ) -> Dict[str, Any]:
        """Fetch wind data for grid points in the bounding box"""
        north, south, east, west = bbox

        # Create grid points
        lats = np.arange(south, north + resolution, resolution)
        lons = np.arange(west, east + resolution, resolution)

        print(f"Fetching wind data for {len(lats)}x{len(lons)} grid points...")

        wind_speeds = np.zeros((len(lats), len(lons)))
        wind_directions = np.zeros((len(lats), len(lons)))

        # Fetch data for each grid point
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                wind_data = self._get_wind_at_point(lat, lon)
                if wind_data:
                    wind_speeds[i, j] = wind_data["speed"]
                    wind_directions[i, j] = wind_data["direction"]

                # Rate limiting for API calls
                if self.api_key:
                    time.sleep(0.1)  # Respect API rate limits

        return {
            "lats": lats,
            "lons": lons,
            "wind_speeds": wind_speeds,
            "wind_directions": wind_directions,
            "bbox": bbox,
        }

    def _get_wind_at_point(self, lat: float, lon: float) -> Optional[Dict[str, float]]:
        """Get wind data at a specific point"""
        if not self.api_key:
            return self._generate_dummy_wind(lat, lon)

        url = f"{self.base_url}/weather"
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                wind_info = data.get("wind", {})

                # Convert m/s to knots and get direction
                speed_ms = wind_info.get("speed", 0)
                speed_knots = speed_ms * 1.944  # Convert m/s to knots
                direction_deg = wind_info.get("deg", 0)  # Degrees from north

                return {"speed": speed_knots, "direction": direction_deg}
            else:
                print(f"API Error for ({lat:.2f}, {lon:.2f}): {response.status_code}")
                return self._generate_dummy_wind(lat, lon)

        except Exception as e:
            print(f"Request error for ({lat:.2f}, {lon:.2f}): {e}")
            return self._generate_dummy_wind(lat, lon)

    def _generate_dummy_wind(self, lat: float, lon: float) -> Dict[str, float]:
        """Generate realistic dummy wind data for testing"""
        # Simulate prevailing winds with geographic variation
        base_direction = 220 + np.sin(lat * 0.05) * 20  # Varies with latitude
        base_speed = 8 + np.sin(lon * 0.1) * 3  # Varies with longitude

        # Add some randomness for realism
        direction = (base_direction + np.random.normal(0, 10)) % 360
        speed = max(1, base_speed + np.random.normal(0, 2))

        return {"speed": speed, "direction": direction}

    def _populate_wind_field(
        self,
        wind_field: "WindField",
        wind_data: Dict[str, Any],
        bbox: Tuple[float, float, float, float],
    ):
        """Convert API wind data to your WindField grid format"""
        north, south, east, west = bbox

        # Get the dimensions of your wind field grid
        height, width = wind_field.height, wind_field.width

        # Create coordinate mappings
        lat_indices = np.linspace(0, len(wind_data["lats"]) - 1, height).astype(int)
        lon_indices = np.linspace(0, len(wind_data["lons"]) - 1, width).astype(int)

        # Populate the wind field grid
        for i in range(height):
            for j in range(width):
                # Get corresponding wind data indices
                lat_idx = lat_indices[i]
                lon_idx = lon_indices[j]

                # Get wind data
                wind_speed = wind_data["wind_speeds"][lat_idx, lon_idx]
                wind_direction_deg = wind_data["wind_directions"][lat_idx, lon_idx]
                wind_direction_rad = radians(wind_direction_deg)

                # Set wind field values (assuming your WindField uses these attributes)
                # You may need to adjust these based on your actual WindField implementation
                if hasattr(wind_field, "wind_speed") and hasattr(
                    wind_field, "wind_direction"
                ):
                    # If your WindField stores speed and direction
                    wind_field.wind_speed[i, j] = wind_speed
                    wind_field.wind_direction[i, j] = wind_direction_rad
                elif hasattr(wind_field, "u") and hasattr(wind_field, "v"):
                    # If your WindField stores u,v components
                    wind_field.u[i, j] = wind_speed * sin(wind_direction_rad)
                    wind_field.v[i, j] = wind_speed * cos(wind_direction_rad)

        print(f"Wind field populated with API data")
        print(
            f"Speed range: {wind_data['wind_speeds'].min():.1f} - {wind_data['wind_speeds'].max():.1f} knots"
        )
        print(f"Average speed: {wind_data['wind_speeds'].mean():.1f} knots")


# Integration function for your existing code
def create_api_wind_field(
    map_size: list,
    bbox: Tuple[float, float, float, float],
    api_key: Optional[str] = None,
) -> "WindField":
    """
    Convenience function to create a WindField with real API data

    Args:
        map_size: [width, height] for your simulation grid
        bbox: (north, south, east, west) geographic bounding box
        api_key: OpenWeatherMap API key (None for dummy data)

    Returns:
        WindField object ready for use with a_star pathfinder
    """
    api_client = WeatherAPIWindField(api_key=api_key)
    return api_client.create_wind_field_from_api(
        width=map_size[0], height=map_size[1], bbox=bbox
    )

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
import rasterio.features
from affine import Affine


class Geography:
    """
    Real-world geography with grid, coastlines, sea mask, and utility functions.
    """

    def __init__(self, start_pos, goal_pos, resolution=1000, epsg_id=3857):
        """
        Args:
            start_pos, goal_pos: Tuples (lon, lat) in degrees
            resolution: grid spacing in meters
            epsg_id: projection EPSG code
        """
        self.resolution = self.dx = self.dy = resolution
        self.epsg_id = epsg_id

        # Compute bounding box from start and goal positions
        self.north, self.south, self.east, self.west = self._compute_bounds(
            start_pos, goal_pos
        )

        # Create ROI polygon in lat/lon
        roi = gpd.GeoDataFrame(
            geometry=[box(self.west, self.south, self.east, self.north)],
            crs="EPSG:4326",
        )
        roi_proj = roi.to_crs(epsg=self.epsg_id)

        # Load shapefiles
        shapefiles = [
            "geography/ne_10m_coastline.shp",
            "geography/ne_10m_land.shp",
            "geography/ne_10m_reefs.shp",
            "geography/ne_10m_minor_islands.shp",
        ]
        layers = [gpd.read_file(f) for f in shapefiles]
        all_layers = pd.concat(layers, ignore_index=True).to_crs(epsg=self.epsg_id)

        # Add this after loading shapefiles but before clipping:
        print(f"Total features before clipping: {len(all_layers)}")
        print(f"All layers bounds: {all_layers.total_bounds}")
        print(f"ROI bounds: {roi_proj.total_bounds}")

        # Clip to ROI
        self.obstacles = gpd.clip(all_layers, roi_proj)

        # Coastlines for plotting
        if "featurecla" in self.obstacles.columns:
            self.coastlines_m = self.obstacles[
                self.obstacles["featurecla"] == "Coastline"
            ]
        else:
            self.coastlines_m = self.obstacles

        print(f"Start position: {start_pos}")
        print(f"Goal position: {goal_pos}")
        print(
            f"ROI bounds (west, south, east, north): {self.west}, {self.south}, {self.east}, {self.north}"
        )

        if self.obstacles.empty:
            # Fallback bounds: use start and goal in projected coordinates
            minx, miny = roi_proj.total_bounds[0], roi_proj.total_bounds[1]
            maxx, maxy = roi_proj.total_bounds[2], roi_proj.total_bounds[3]
        else:
            minx, miny, maxx, maxy = self.obstacles.total_bounds

        # Compute bounds in meters
        # minx, miny, maxx, maxy = self.obstacles.total_bounds
        self.bounds_m = (minx, maxx, miny, maxy)
        self.minx, self.maxx, self.miny, self.maxy = self.bounds_m

        # Create grid
        x = np.arange(minx, maxx + resolution, resolution)
        y = np.arange(miny, maxy + resolution, resolution)
        self.xx, self.yy = np.meshgrid(x, y)

        # Rasterize sea mask (1 = sea, 0 = land/obstacle)
        transform = Affine.translation(minx, miny) * Affine.scale(
            resolution, resolution
        )
        mask_raster = rasterio.features.rasterize(
            [(geom, 1) for geom in self.obstacles.geometry],
            out_shape=self.xx.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        self.sea_mask = mask_raster == 0

        # Projection transformer (lat/lon <-> meters)
        self.transformer = Transformer.from_crs(
            "EPSG:4326", self.epsg_id, always_xy=True
        )
        self.inv_transformer = Transformer.from_crs(
            self.epsg_id, "EPSG:4326", always_xy=True
        )

    @staticmethod
    def _compute_bounds(pos1, pos2):
        """Compute (north, south, east, west) from two positions"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        north = max(lat1, lat2)
        south = min(lat1, lat2)
        east = max(lon1, lon2)
        west = min(lon1, lon2)
        return north, south, east, west

    def snap_to_sea(self, x, y):
        """
        Snap coordinates to nearest sea grid cell.
        """
        mask = self.sea_mask
        x_flat = self.xx[mask]
        y_flat = self.yy[mask]
        dists = (x_flat - x) ** 2 + (y_flat - y) ** 2
        idx = np.argmin(dists)
        # Clamp to bounds
        sx = np.clip(x_flat[idx], self.minx, self.maxx)
        sy = np.clip(y_flat[idx], self.miny, self.maxy)
        return sx, sy

    def is_sea(self, x, y):
        """
        Check if a coordinate is sea.
        """
        ix = int((x - self.minx) / self.resolution)
        iy = int((y - self.miny) / self.resolution)
        if 0 <= iy < self.sea_mask.shape[0] and 0 <= ix < self.sea_mask.shape[1]:
            return self.sea_mask[iy, ix]
        return False

    def meters_to_geo(self, x, y):
        """Convert meter coordinates to lat/lon"""
        return self.inv_transformer.transform(x, y)

    def geo_to_meters(self, lon, lat):
        """Convert lat/lon to meters"""
        return self.transformer.transform(lon, lat)

    def meters_to_index(self, x, y):
        ix = int((x - self.minx) / self.resolution)
        iy = int((y - self.miny) / self.resolution)
        ix = np.clip(ix, 0, self.sea_mask.shape[1] - 1)
        iy = np.clip(iy, 0, self.sea_mask.shape[0] - 1)
        return ix, iy

    def index_to_meters(self, ix, iy):
        x = self.minx + ix * self.resolution
        y = self.miny + iy * self.resolution
        return x, y

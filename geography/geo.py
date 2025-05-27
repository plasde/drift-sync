import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import box, Point
from shapely.prepared import prep
from pyproj import Transformer
import rasterio.features
from affine import Affine


class Geo:
    def __init__(self, dx=1000, dy=1000, epsg_id=28992):

        self.epsg_id = epsg_id
        self.dx = dx
        self.dy = dy

        # Define region
        lat_min, lat_max = 52.85, 53.25
        lon_min, lon_max = 4.65, 5.55
        roi = gpd.GeoDataFrame(geometry=[box(lon_min, lat_min, lon_max, lat_max)], crs="EPSG:4326")
        roi_proj = roi.to_crs(epsg=self.epsg_id)

        # Load geometry
        names = ['ne_10m_coastline', 'ne_10m_land', 'ne_10m_reefs', 'ne_10m_minor_islands']
        layers = [gpd.read_file(f'geography/{name}.shp') for name in names]
        all_layers = pd.concat(layers, ignore_index=True).to_crs(epsg=self.epsg_id)
        self.obstacles = gpd.clip(all_layers, roi_proj)
        self.coastlines_m = self.obstacles[self.obstacles['featurecla'] == 'Coastline']

        # Grid
        self.minx, self.miny, self.maxx, self.maxy = self.obstacles.total_bounds
        x = np.arange(self.minx, self.maxx, dx)
        y = np.arange(self.miny, self.maxy, dy)
        self.xx, self.yy = np.meshgrid(x, y)

        transform = Affine.translation(self.minx, self.miny) * Affine.scale(dx, dy)
        mask_raster = rasterio.features.rasterize(
            [(geom, 1) for geom in self.obstacles.geometry],
            out_shape=self.xx.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8'
        )
        self.sea_mask = mask_raster == 0
        self.transformer = Transformer.from_crs("EPSG:4326", self.epsg_id, always_xy=True)

    def snap_to_sea(self, x, y):
        mask = self.sea_mask
        x_flat = self.xx[mask]
        y_flat = self.yy[mask]
        dists = (x_flat - x)**2 + (y_flat - y)**2
        idx = np.argmin(dists)
        return x_flat[idx], y_flat[idx]

    def is_sea(self, x, y):
        ix = int((x - self.minx) / self.dx)
        iy = int((y - self.miny) / self.dy)
        if 0 <= iy < self.sea_mask.shape[0] and 0 <= ix < self.sea_mask.shape[1]:
            return self.sea_mask[iy, ix]
        return False  # Outside grid

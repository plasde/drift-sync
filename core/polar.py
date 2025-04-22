from math import pi, radians

# Boat polar: simplistic VMG model (angle vs speed %)
def polar_performance(apparent_wind_angle_rad):
    angle = abs((apparent_wind_angle_rad + pi) % (2 * pi) - pi)  # Normalize between 0 and pi
    if angle < radians(30):  # Too close to wind (no-go zone)
        return 0.05
    elif angle < radians(45):
        return 0.15
    elif angle < radians(60):
        return 0.5
    elif angle < radians(75):
        return 0.8
    elif angle < radians(90):
        return 1.2  # build up speed
    elif angle < radians(105):
        return 0.8
    elif angle < radians(120):
        return 0.5
    elif angle <= radians(150):
        return 0.3
    else:
        return 0.2  # dead downwind slower

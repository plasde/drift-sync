from math import pi, radians

# Boat polar: simplistic VMG model (angle vs speed %)
def polar_performance(apparent_wind_angle_rad):
    angle = abs((apparent_wind_angle_rad + pi) % (2 * pi) - pi)  # Normalize between 0 and pi
    if angle < radians(30):  # Too close to wind (no-go zone)
        return 0.05
    elif angle < radians(40):
        return 0.2
    elif angle < radians(90):
        return (angle - radians(40)) / radians(50) * 0.8 + 0.2  # build up speed
    elif angle <= radians(150):
        return 1.0
    else:
        return 0.6  # dead downwind slower

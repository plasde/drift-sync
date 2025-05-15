from math import pi, radians, degrees

def polar_performance(apparent_wind_angle_rad):
    angle = apparent_wind_angle_rad #abs((apparent_wind_angle_rad + pi) % (2 * pi) - pi)  # Normalize between 0 and pi

    if angle < radians(30):
        return 0.0  # No-go zone

    # Smooth curve: use piecewise linear interpolation
    # You can later replace with spline if needed

    # Performance keypoints (angle in degrees, speed multiplier)
    points = [
        (30, 0.0),
        (45, 0.5),
        (60, 0.9),
        (75, 1.1),
        (90, 1.3),
        (105, 1.1),
        (120, 0.9),
        (150, 0.5),
        (180, 0.3)
    ]

    # Linear interpolation between points
    degrees_angle = degrees(angle)
    for i in range(len(points) - 1):
        a0, v0 = points[i]
        a1, v1 = points[i+1]
        if a0 <= degrees_angle <= a1:
            # Interpolate
            t = (degrees_angle - a0) / (a1 - a0)
            return v0 + t * (v1 - v0)

    # If beyond 180°, return dead downwind value
    return 0.3

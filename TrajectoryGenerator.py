"""
Trajectory Generator (WGS84 Earth)
"""
import pyvista as pv
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image  # Required for texture mapping

# =============================================================================
#                               Settings
# =============================================================================

# Latitude / Longitude / Altitude Waypoints (Degrees, Degrees, Meters)
target_waypoint_lla = np.array([
    [-45,  0, 50e3],
    [ 00,  0, 200e3],
    [ 45,  0, 50e3]
])

# Aircraft velocity in m/s (500 km/h converted to m/s)
velocity_m_per_sec = (500) * 1000 / 3600

# Interpolation Settings
points_to_interpolate = 100

# Plot and Save Settings
enable_plot = True
enable_save = True
save_file_name = "Trajectory.txt"

# Choose output unit for Euler Angles: 'degrees' or 'radians'
euler_unit = 'degrees'

# =============================================================================
#                        WGS84 Ellipsoid Constants
# =============================================================================

semi_major_axis_a = 6378137               # Semi-major axis in meters
flattening_f = 1 / 298.257223563          # Flattening factor
eccentricity_sq_e2 = 2 * flattening_f - flattening_f**2 # First eccentricity squared

# =============================================================================
#                        Waypoint Interpolation
# =============================================================================

# Generate original waypoint indices and the new interpolated indices
num_waypoints = len(target_waypoint_lla)
waypoint_indices = np.arange(1, num_waypoints + 1)
interpolated_indices = np.linspace(1, num_waypoints, points_to_interpolate)

# Perform linear interpolation for Lat, Lon, and Alt using NumPy
lat_interpolated = np.interp(interpolated_indices, waypoint_indices, target_waypoint_lla[:, 0])
lon_interpolated = np.interp(interpolated_indices, waypoint_indices, target_waypoint_lla[:, 1])
alt_interpolated = np.interp(interpolated_indices, waypoint_indices, target_waypoint_lla[:, 2])

# Convert Geographic coordinates (Lat/Lon) to radians
lat_rad = np.radians(lat_interpolated)
lon_rad = np.radians(lon_interpolated)
alt = alt_interpolated

# =============================================================================
#                Geodetic (LLA) to Cartesian (ECEF) Conversion
# =============================================================================

# Calculate the prime vertical radius of curvature (N)
N = semi_major_axis_a / np.sqrt(1 - eccentricity_sq_e2 * np.sin(lat_rad)**2)

# Convert LLA to ECEF coordinates using WGS84 ellipsoidal geometry
x_ecef = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
y_ecef = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
z_ecef = ((1 - eccentricity_sq_e2) * N + alt) * np.sin(lat_rad)

traj_length = len(x_ecef)
time_array = np.zeros(traj_length)

# Preallocate kinematics arrays
vx, vy, vz = np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length)
ax, ay, az = np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length)

# =============================================================================
#                Kinematics: Velocity and Acceleration
# =============================================================================

# Compute time differences and velocity components
for i in range(1, traj_length):
    dx = x_ecef[i] - x_ecef[i-1]
    dy = y_ecef[i] - y_ecef[i-1]
    dz = z_ecef[i] - z_ecef[i-1]
    
    total_distance = math.sqrt(dx**2 + dy**2 + dz**2)
    dt = total_distance / velocity_m_per_sec
    
    vx[i], vy[i], vz[i] = dx / dt, dy / dt, dz / dt
    time_array[i] = time_array[i-1] + dt

# Boundary condition: Set initial velocity equal to the second point
vx[0], vy[0], vz[0] = vx[1], vy[1], vz[1]

# Calculate Accelerations (Finite differences of velocity over time)
for i in range(1, traj_length):
    dt_step = time_array[i] - time_array[i-1]
    ax[i] = (vx[i] - vx[i-1]) / dt_step
    ay[i] = (vy[i] - vy[i-1]) / dt_step
    az[i] = (vz[i] - vz[i-1]) / dt_step

# Boundary condition: Set initial acceleration equal to the second point
ax[0], ay[0], az[0] = ax[1], ay[1], az[1]

# =============================================================================
#                Attitude: Euler Angles and Quaternions
# =============================================================================

yaw_array, pitch_array, roll_array = np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length)
q0_array, q1_array, q2_array, q3_array = np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length)

for i in range(traj_length):
    lat_i = lat_rad[i]
    lon_i = lon_rad[i]
    
    # Direction Cosine Matrix from ECEF to local NED (North-East-Down)
    R_ECEF2NED = np.array([
        [-math.sin(lat_i)*math.cos(lon_i), -math.sin(lat_i)*math.sin(lon_i),  math.cos(lat_i)],
        [-math.sin(lon_i),                  math.cos(lon_i),                  0              ],
        [-math.cos(lat_i)*math.cos(lon_i), -math.cos(lat_i)*math.sin(lon_i), -math.sin(lat_i)]
    ])

    # Transform ECEF velocity to local NED frame
    v_ecef = np.array([vx[i], vy[i], vz[i]])
    v_ned  = R_ECEF2NED.dot(v_ecef)
    vN, vE, vD = v_ned[0], v_ned[1], v_ned[2]
    
    # Extract Euler Angles
    yaw_rad   = math.atan2(vE, vN)
    pitch_rad = math.atan2(-vD, math.sqrt(vN**2 + vE**2))
    roll_rad  = 0.0 # Wings level assumption
    
    if euler_unit == 'degrees':
        yaw_array[i], pitch_array[i], roll_array[i] = math.degrees(yaw_rad), math.degrees(pitch_rad), math.degrees(roll_rad)
    else:
        yaw_array[i], pitch_array[i], roll_array[i] = yaw_rad, pitch_rad, roll_rad
    
    # Scalar-first Quaternions
    q0_array[i] = math.cos(yaw_rad/2)*math.cos(pitch_rad/2)*math.cos(roll_rad/2) + math.sin(yaw_rad/2)*math.sin(pitch_rad/2)*math.sin(roll_rad/2)
    q1_array[i] = math.cos(yaw_rad/2)*math.cos(pitch_rad/2)*math.sin(roll_rad/2) - math.sin(yaw_rad/2)*math.sin(pitch_rad/2)*math.cos(roll_rad/2)
    q2_array[i] = math.cos(yaw_rad/2)*math.sin(pitch_rad/2)*math.cos(roll_rad/2) + math.sin(yaw_rad/2)*math.cos(pitch_rad/2)*math.sin(roll_rad/2)
    q3_array[i] = math.sin(yaw_rad/2)*math.cos(pitch_rad/2)*math.cos(roll_rad/2) - math.cos(yaw_rad/2)*math.sin(pitch_rad/2)*math.sin(roll_rad/2)


trajectory = {
    'time': time_array, 
    'lat': lat_interpolated, 'lon': lon_interpolated, 'alt': alt_interpolated,
    'x': x_ecef, 'y': y_ecef, 'z': z_ecef,
    'vx': vx, 'vy': vy, 'vz': vz, 'ax': ax, 'ay': ay, 'az': az,
    'yaw': yaw_array, 'pitch': pitch_array, 'roll': roll_array,
    'q0': q0_array, 'q1': q1_array, 'q2': q2_array, 'q3': q3_array
}

# =============================================================================
#                                Plotting
# =============================================================================

if enable_plot:
    semi_major_axis_a = 6378137.0
    semi_minor_axis_b = 6356752.3142
    
    plotter = pv.Plotter()
    earth = pv.ParametricEllipsoid(semi_major_axis_a, semi_major_axis_a, semi_minor_axis_b, u_res=400, v_res=200)
    earth = earth.texture_map_to_sphere(prevent_seam = False, inplace=False)
    
    texture = pv.read_texture('EarthTexture.jpg')
    texture = texture.flip_y()
    
    plotter.add_mesh(earth, texture=texture, smooth_shading=True)    
    
    trajectory_points = np.column_stack((trajectory['x'], trajectory['y'], trajectory['z']))
    trajectory_line = pv.lines_from_points(trajectory_points)
    plotter.add_mesh(trajectory_line, color="red", line_width=5)
    
    plotter.set_background("black")
    plotter.show(full_screen=True)

# =============================================================================
#                               File Saving
# =============================================================================

if enable_save:

    field_names = ['time', 'lat', 'lon', 'alt', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 
                   'yaw', 'pitch', 'roll', 'q0', 'q1', 'q2', 'q3']
    column_names = ['Time', 'Lat', 'Lon', 'Alt', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 
                    'yaw', 'pitch', 'roll', 'q0', 'q1', 'q2', 'q3']
    num_cols = len(field_names)
    W = np.zeros(num_cols, dtype=int)
    
    for k in range(num_cols):
        col_data = trajectory[field_names[k]]
        str_lengths = [len(f"{val:.8f}") for val in col_data]
        min_L = min(str_lengths)
        if k == 0:
            W[k] = max(len(column_names[k]), max(str_lengths))
        else:
            W[k] = min_L + 12

    try:
        with open(save_file_name, 'w') as fid:
            header_row = "".join([f"{column_names[k]:>{W[k]}s}" for k in range(num_cols)])
            fid.write(header_row + '\n')
            for i in range(traj_length):
                data_row = ""
                for k in range(num_cols):
                    col_data = trajectory[field_names[k]]
                    data_row += f"{col_data[i]:>{W[k]}.8f}"
                fid.write(data_row + '\n')
        print(f"Trajectory successfully exported to: {save_file_name}")
    except IOError:
        print("File could not be opened. Check folder permissions.")
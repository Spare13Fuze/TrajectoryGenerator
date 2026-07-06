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

# Latitude / Longitude / Altitude / Speed Waypoints (Degrees, Degrees, Meters, Km/h)
target_waypoint_lla_speed = np.array([
    [-45.0,  0,  50e3,  400],
    [-22.5, -5,  500e3,  400],
    [ 00.0,  0, 950e3,  900],
    [ 22.5,  5,  500e3,  300],
    [ 45.0,  0,  50e3,  300]
])

# Maximum Bank angle that the body is capable of achieving
max_bank_angle_degs = 30

# Interpolation Settings
points_to_interpolate = 100

# Plot and Save Settings
enable_plot = True
enable_save = False
save_file_name = "Trajectory.txt"

# Choose output unit for Euler Angles: 'degrees' or 'radians'
euler_unit = 'degrees'

# =============================================================================
#                        Constants
# =============================================================================

# WGS84 Constants
semi_major_axis_a = 6378137.0             # Semi-major axis in meters
semi_minor_axis_b = 6356752.3142          # Semi-minor axis in meters
flattening_f = 1 / 298.257223563          # Flattening factor
eccentricity_sq_e2 = 2 * flattening_f - flattening_f**2 # First eccentricity squared

# Gravity Acceleration
g = 9.81 

# =============================================================================
#                        Waypoint Interpolation
# =============================================================================

# Generate original waypoint indices and the new interpolated indices
num_waypoints = len(target_waypoint_lla_speed)
waypoint_indices = np.arange(1, num_waypoints + 1)
interpolated_indices = np.linspace(1, num_waypoints, points_to_interpolate)

lat_interpolated = np.interp(interpolated_indices, waypoint_indices, target_waypoint_lla_speed[:, 0])
lon_interpolated = np.interp(interpolated_indices, waypoint_indices, target_waypoint_lla_speed[:, 1])
alt_interpolated = np.interp(interpolated_indices, waypoint_indices, target_waypoint_lla_speed[:, 2])

# Interpolate the speed profile across the path
speed_profile_kmh = np.interp(interpolated_indices, waypoint_indices, target_waypoint_lla_speed[:, 3])
speed_profile_m_s = speed_profile_kmh * 1000 / 3600

# Convert Geographic coordinates (Lat/Lon) to radians
lat_rad = np.radians(lat_interpolated)
lon_rad = np.radians(lon_interpolated)
alt = alt_interpolated

# =============================================================================
#                Geodetic (LLA) to Cartesian (ECR) Conversion
# =============================================================================

# Calculate the prime vertical radius of curvature (N)
N = semi_major_axis_a / np.sqrt(1 - eccentricity_sq_e2 * np.sin(lat_rad)**2)

# Convert LLA to ECR coordinates using WGS84 ellipsoidal geometry
x_ecr = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
y_ecr = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
z_ecr = ((1 - eccentricity_sq_e2) * N + alt) * np.sin(lat_rad)

traj_length = len(x_ecr)
time_array = np.zeros(traj_length)

# =============================================================================
#                Kinematics: Velocity and Acceleration
# =============================================================================

# Preallocate arrays for Velocity & Acceleration
vx, vy, vz = np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length)
ax, ay, az = np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length)

# Compute time difference between timesteps, then velocity components
for i in range(1, traj_length):
    dx = x_ecr[i] - x_ecr[i-1]
    dy = y_ecr[i] - y_ecr[i-1]
    dz = z_ecr[i] - z_ecr[i-1]
    
    total_distance = math.sqrt(dx**2 + dy**2 + dz**2)
    
    # Use the specific interpolated speed for this segment
    current_speed = speed_profile_m_s[i]
    dt = total_distance / current_speed if current_speed > 0 else 0.1
    
    vx[i], vy[i], vz[i] = dx / dt, dy / dt, dz / dt
    time_array[i] = time_array[i-1] + dt

# Boundary condition: Set initial velocity equal to the second point
vx[0], vy[0], vz[0] = vx[1], vy[1], vz[1]

# Calculate Accelerations
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

# Preallocate arrays for Euler Angles and Quaternions
yaw_array, pitch_array, roll_array = np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length)
q0_array, q1_array, q2_array, q3_array = np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length), np.zeros(traj_length)

# Calculate local NED velocities, Yaw, and Pitch
for i in range(traj_length):
    lat_i = lat_rad[i]
    lon_i = lon_rad[i]
    
    R_ecr2NED = np.array([
        [-math.sin(lat_i)*math.cos(lon_i), -math.sin(lat_i)*math.sin(lon_i),  math.cos(lat_i)],
        [-math.sin(lon_i),                  math.cos(lon_i),                  0              ],
        [-math.cos(lat_i)*math.cos(lon_i), -math.cos(lat_i)*math.sin(lon_i), -math.sin(lat_i)]
    ])

    v_ecr = np.array([vx[i], vy[i], vz[i]])
    v_ned  = R_ecr2NED.dot(v_ecr)
    vN, vE, vD = v_ned[0], v_ned[1], v_ned[2]
    
    yaw_array[i]   = math.atan2(vE, vN)
    pitch_array[i] = math.atan2(-vD, math.sqrt(vN**2 + vE**2))

# Calculate Roll based on rate of turn (yaw rate)
for i in range(traj_length):
    
    if i == 0:
        yaw_rate = (yaw_array[1] - yaw_array[0]) / (time_array[1] - time_array[0])
    elif i == traj_length - 1:
        yaw_rate = (yaw_array[i] - yaw_array[i-1]) / (time_array[i] - time_array[i-1])
    else:
        # Centered difference for smoother rate calculation
        yaw_rate = (yaw_array[i+1] - yaw_array[i-1]) / (time_array[i+1] - time_array[i-1])
    
    # Handle angle wrapping issues (-pi to pi jump)
    if yaw_rate > math.pi: yaw_rate -= 2 * math.pi
    if yaw_rate < -math.pi: yaw_rate += 2 * math.pi
        
    # Ground speed in the local horizontal plane
    v_horiz = math.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2) 
    
    # Aircraft Coordinated Turn Equation: tan(Roll) = (Velocity * Yaw_Rate) / g
    roll_array[i] = math.atan2(v_horiz * yaw_rate, g)
    
    max_bank = math.radians(max_bank_angle_degs)
    roll_array[i] = np.clip(roll_array[i], -max_bank, max_bank)

# Convert arrays to output settings and compute Quaternions
for i in range(traj_length):
    y_r, p_r, r_r = yaw_array[i], pitch_array[i], roll_array[i]
    
    if euler_unit == 'degrees':
        yaw_array[i], pitch_array[i], roll_array[i] = math.degrees(y_r), math.degrees(p_r), math.degrees(r_r)
    else:
        yaw_array[i], pitch_array[i], roll_array[i] = y_r, p_r, r_r
        
    # Re-calculate Quaternions with the newly introduced roll component
    q0_array[i] = math.cos(y_r/2)*math.cos(p_r/2)*math.cos(r_r/2) + math.sin(y_r/2)*math.sin(p_r/2)*math.sin(r_r/2)
    q1_array[i] = math.cos(y_r/2)*math.cos(p_r/2)*math.sin(r_r/2) - math.sin(y_r/2)*math.sin(p_r/2)*math.cos(r_r/2)
    q2_array[i] = math.cos(y_r/2)*math.sin(p_r/2)*math.cos(r_r/2) + math.sin(y_r/2)*math.cos(p_r/2)*math.sin(r_r/2)
    q3_array[i] = math.sin(y_r/2)*math.cos(p_r/2)*math.cos(r_r/2) - math.cos(y_r/2)*math.sin(p_r/2)*math.sin(r_r/2)

trajectory = {
    'time': time_array, 
    'lat': lat_interpolated, 'lon': lon_interpolated, 'alt': alt_interpolated,
    'x': x_ecr, 'y': y_ecr, 'z': z_ecr,
    'vx': vx, 'vy': vy, 'vz': vz, 'ax': ax, 'ay': ay, 'az': az,
    'yaw': yaw_array, 'pitch': pitch_array, 'roll': roll_array,
    'q0': q0_array, 'q1': q1_array, 'q2': q2_array, 'q3': q3_array
}

# =============================================================================
#                                Plotting
# =============================================================================

if enable_plot:
    
    plotter = pv.Plotter()
    earth = pv.ParametricEllipsoid(semi_major_axis_a, semi_major_axis_a, semi_minor_axis_b, u_res=800, v_res=400)
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
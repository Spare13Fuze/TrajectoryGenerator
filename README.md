# Trajectory Simulator

Generate, Plot, Save and Load Object trajectories over a high-fidelity Earth model.

---

## Components

* **`TrajectoryGenerator.py`**: Synthesises a detailed flight path from geographic waypoints, computes kinematic and attitude states, renders a 3D globe visualisation, and exports the data to a text file.


* **`TrajectoryReader.py`**: Provides a flexible utility function to load the exported text data back into Python, featuring customizable column filters to streamline data ingestion.


---

## System Requirements & Dependencies

* Python 3 


Libraries:

* **NumPy**: For vectorized mathematical operations and array handling.


* **PyVista**: For 3D geometric rendering and ellipsoid texture mapping.


* **Matplotlib**: Embedded dependency for coordinate projection utilities.


* **Pillow (PIL)**: Required by PyVista to parse and flip the Earth texture image.


---

## Usage Example

### Step 1: Generate the Trajectory

Run the generator script to compute the flight profiles, view the 3D globe rendering, and write the trajectory data to disk:

```bash
python TrajectoryGenerator.py

```

### Step 2: Read and Filter the Data

Utilize the parser function inside your downstream analysis scripts to read specific telemetry subsets:

```python
from TrajectoryReader import load_trajectory_data

# Load only Cartesian positions and Euler attitude metrics from the data log
trajectory = load_trajectory_data('Trajectory.txt', pos_switch='ECEF', att_switch='EULER')

# Access the data cleanly via dictionary headers
print("Simulated Time Steps:", trajectory['Time'])
print("X Coordinates (ECEF):", trajectory['x'])
print("Calculated Aircraft Yaw:", trajectory['yaw'])

```


---

## Text File Structure Matrix

When exported by `TrajectoryGenerator.py`, the resulting `Trajectory.txt` layout maps directly to the following matrix columns:

| Column Category | Associated Header Variables |
| --- | --- |
| **Time** | `Time`<br> |
| **Geographic Coordinates (LLA)** | `Lat`, `Lon`, `Alt`<br> |
| **Cartesian Coordinates (ECEF)** | `x`, `y`, `z`<br> |
| **Kinematic Velocities** | `vx`, `vy`, `vz`<br> |
| **Kinematic Accelerations** | `ax`, `ay`, `az`<br> |
| **Attitude (Euler Angles)** | `yaw`, `pitch`, `roll`<br> |
| **Attitude (Quaternions)** | `q0`, `q1`, `q2`, `q3`<br> |
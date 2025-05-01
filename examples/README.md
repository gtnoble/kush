# Peridynamics Examples

This directory contains example configuration files for running peridynamics simulations.

## 2D Plate Example

A simple 2D plate example with four points:
- Bottom-left point: Fixed velocity (anchored)
- Top-right point: Applied upward force
- Other points: Free to move

### Configuration Files
- `2d_plate.materials.json`: Material properties (aluminum)
- `2d_plate.simulation.json`: Simulation parameters including time stepping and output settings
- `2d_plate.points.jsonl`: Point configurations (positions, forces, constraints)

### Running the Example
```bash
# From project root
./peridynamics \
  -d 2 \
  --points examples/2d_plate.points.jsonl \
  --materials examples/2d_plate.materials.json \
  --simulation examples/2d_plate.simulation.json
```

The simulation will run and output results to `2d_plate_result.csv`.

## File Formats

### Materials (JSON)
```json
{
  "materials": {
    "material_name": {
      "density": 2700.0,            // kg/m³
      "youngsModulus": 70e9,        // Pa
      "criticalStretch": 10,        // []
      "velocity": [vx, vy],         // Optional: Initial velocity
      "fixed_velocity": false,      // Optional: If true, velocity remains constant
      "force": [fx, fy],           // Optional: Force applied to all points
      "ramp_duration": 1e-6        // Optional: Force ramp duration (s)
    }
  }
}
```

### Points (JSONL)
Each line is a JSON object:
```json
{
  "position": [x, y],           // Position coordinates
  "material": "material_name",  // Reference to material
  "thickness": value,          // Required for 1D/2D points
  "velocity": [vx, vy],        // Optional: Initial velocity
  "fixed_velocity": true,      // Optional: If true, velocity remains constant
  "force": [fx, fy],          // Optional: Applied force
  "ramp_duration": 1e-6       // Optional: Force ramp duration (s)
}
```

### Simulation (JSON)
```json
{
  "horizon": 0.003,            // Horizon radius (m)
  "optimization": {
    "tolerance": 1e-6,         // Convergence tolerance
    "max_iterations": 10,      // Maximum iterations per step
    "momentum": 0.1,           // Momentum term [0,1)
    "gradient_step_size": {
      "value": 1e-4,           // Fixed step size
      "horizon_fraction": 0.0001 // Optional: step = horizon * fraction
    }
  },
  "time_stepping": {
    "initial_step": 1e-6,      // Optional: Initial time step
    "max_step": 1e-4,         // Maximum allowed time step
    "total_time": 3e-3,       // Total simulation time
    "safety_factor": 0.1,      // Time step safety factor (0,1]
    "characteristic_velocity": 1.0, // Optional: Override material wave speed
    "max_relative_motion": 0.1,    // Maximum relative point motion
    "response_lag": 1000          // Time step adjustment lag
  },
  "damping": {
    "mass_time_constant": 1e-4,         // Mass damping time constant
    "viscosity_time_constant": 1e-4     // Optional: Viscosity damping
  },
  "output": {
    "csv_file": "result.csv",          // Output file path
    "step_interval": 100               // Save every N steps
    // OR
    "time_interval": 1e-4              // Save every T seconds
  }
}

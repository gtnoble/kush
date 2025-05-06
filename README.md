# Peridynamics Simulation Framework

A flexible, type-safe implementation of peridynamics in the D programming language that supports multiple dimensions and different material models.

## Features

- Support for 1D, 2D, and 3D simulations
- Configuration-based simulation setup
- Bond-based peridynamic model
- Adaptive time stepping
- Multiple optimization methods:
  - Gradient descent with momentum
  - Parallel tempering for non-convex problems
- Line-delimited JSON for point configurations
- CSV output for analysis

## Building

```bash
# Build with DUB
dub build
```

## Running Simulations

```bash
# Basic usage
peridynamics -d <dimension> -p <points> -m <materials> -s <simulation>

# Example: 2D plate simulation
peridynamics \
  -d 2 \
  --points examples/2d_plate.points.jsonl \
  --materials examples/2d_plate.materials.json \
  --simulation examples/2d_plate.simulation.json

# Show help
peridynamics --help
```

## Configuration Files

### Points (JSONL)
Each line defines a point:
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

### Simulation (JSON)
```json
{
  "horizon": 0.003,            // Horizon radius (m)
  "optimization": {
    "solver_type": "gradient_descent",  // "gradient_descent" or "parallel_tempering"
    "tolerance": 1e-6,                  // Convergence tolerance
    "max_iterations": 10,               // Maximum iterations per step
    "momentum": 0.9,                    // Momentum term [0,1)
    "gradient_mode": "step_size",       // "step_size" (default) or "learning_rate"
    "learning_rate": 0.01,             // Learning rate (for gradient descent)
    "gradient_step_size": {
      "value": 1e-4,                   // Fixed step size (for gradient descent)
      "horizon_fraction": 0.0001       // Optional: step = horizon * fraction
    },
    "parallel_tempering": {            // Only used when solver_type is "parallel_tempering"
      "num_replicas": 8,              // Optional: defaults to CPU core count
      "num_processes": 4,             // Optional: defaults to CPU core count
      "min_temperature": 0.1,         // Controls local optimization (cold replicas)
      "max_temperature": 2.0          // Controls global exploration (hot replicas)
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
    "step_interval": 100,              // Save every N steps
    // OR
    "time_interval": 1e-4              // Save every T seconds
  }
}
```

## Examples

The `examples/` directory contains sample configurations:
- `2d_plate.*`: A 2D plate with an applied force
- See `examples/README.md` for details

## Testing

The `test/` directory contains test configurations:
- `1d_test.*`: Simple 1D rod test
- `3d_test.*`: 3D cube corner test
- See `test/README.md` for details

## Output Format

The simulation produces CSV files with:
- Point positions over time
- Velocities
- Forces
- Damage state

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - See LICENSE file for details

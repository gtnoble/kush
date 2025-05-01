# Peridynamics Test Configurations

This directory contains test configurations for verifying the peridynamics simulation framework.

## Test Cases

### 1. 1D Rod (configs/1d_test.*)
A simple rod with 5 points:
- One end fixed (left)
- Tensile force applied to other end (right)
- Linear arrangement of points
- Tests 1D behavior and thickness requirements

```bash
./peridynamics \
  -d 1 \
  --points test/configs/1d_test.points.jsonl \
  --materials test/configs/1d_test.materials.json \
  --simulation test/configs/1d_test.simulation.json
```

### 2. 3D Cube Corner (configs/3d_test.*)
A cube with 8 points:
- One corner fixed (origin)
- Diagonal force applied to opposite corner
- Tests 3D behavior and force distribution
- Demonstrates 3D deformation

```bash
./peridynamics \
  -d 3 \
  --points test/configs/3d_test.points.jsonl \
  --materials test/configs/3d_test.materials.json \
  --simulation test/configs/3d_test.simulation.json
```

## Output Files
Each test will generate a CSV file with results:
- `1d_test_result.csv`: 1D rod simulation results
- `3d_test_result.csv`: 3D cube simulation results

The CSV files contain:
- Point positions over time
- Velocities
- Forces
- Damage state

## Configuration Features Demonstrated

### Points
- Fixed velocity constraints
- Force application
- 1D thickness requirements
- Initial positioning

### Materials
- Different material properties (steel, titanium)
- Density and elastic constants
- Critical stretch values

### Simulation
- Different output methods (step vs time interval)
- Horizon size appropriate for dimensions
- Time stepping parameters
- Optimization settings

## Running Tests
Each test can be run independently. The output CSV files can be analyzed to verify:
1. Conservation of momentum
2. Energy behavior
3. Force distribution
4. Deformation patterns

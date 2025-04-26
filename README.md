# Generic Peridynamics Framework in D

A flexible, generic implementation of the peridynamic model in D, supporting 1D, 2D, and 3D simulations with reversible damage mechanics.

## Features

- Generic vector implementation supporting 1D, 2D, and 3D spaces
- Template-based material point interface for custom material models
- Bond-based peridynamics implementation with reversible damage
- Dynamic neighbor search based on current positions
- Type-safe design using D's powerful template system

## Requirements

- D compiler (DMD, LDC, or GDC)
- DUB package manager

## Building

```bash
# Debug build
dub build

# Release build
dub build -b release
```

## Running the Example

The included example simulates a 2D plate with a pre-crack under mode I loading:

```bash
dub run
```

The simulation outputs the final positions of material points to stdout in CSV format (x,y).

## Framework Structure

```
src/
├── app.d                  # Example application
├── math/
│   └── vector.d          # Generic vector implementation
├── core/
│   ├── material_point.d  # Generic material point interface
│   ├── material_body.d   # Material body container
│   └── simulation.d      # Simulation engine
└── models/
    └── bond_based_point.d # Bond-based peridynamics implementation
```

## Creating Custom Material Models

To implement a custom material model:

1. Create a new class that implements `MaterialPoint!(T, V)`
2. Implement required properties: `position` and `referencePosition`
3. Implement arithmetic operators: `opBinary("+")` and `opBinary("*")`
4. Implement the `stateRate` method with your material model

Example:

```d
class CustomPoint(V) : MaterialPoint!(CustomPoint!V, V)
    if (isVector!V)
{
    // Implement required interface...
}
```

## Example Usage

```d
// Create material points
auto points = new BondBasedPoint!Vector2D[];
// ... initialize points ...

// Create material body
auto body = new MaterialBody!(BondBasedPoint!Vector2D, Vector2D)(points, horizon);

// Run simulation
simulate(body, timeStep, numSteps);
```

## License

This project is available under the MIT License.

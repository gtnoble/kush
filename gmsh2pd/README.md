# GMSH to Peridynamics Points Converter

A utility to convert GMSH mesh files into point data for peridynamics simulations.

## Features

- Supports both 2D and 3D meshes
- Converts triangular and tetrahedral elements
- Preserves physical group assignments as material groups
- Calculates centroids and volumes automatically
- JSONL output format compatible with peridynamics solver

## Requirements

- D compiler (DMD, LDC, or GDC)
- GMSH development files
- DUB package manager

## Building

```bash
dub build
```

## Usage

Basic usage:
```bash
gmsh2pd -i input.msh -o output.jsonl
```

Options:
- `-i, --input`: Input GMSH mesh file (.msh)
- `-o, --output`: Output points file (.jsonl)
- `-v, --validate`: Validate output points (optional)

## Output Format

For 2D meshes:
```jsonl
{"position": [x, y], "volume": area, "material_groups": ["group1", ...]}
```

For 3D meshes:
```jsonl
{"position": [x, y, z], "volume": volume, "material_groups": ["group1", ...]}
```

## Notes

- 2D meshes should use triangular elements
- 3D meshes should use tetrahedral elements
- Physical groups in the GMSH mesh are converted to material groups
- Each element must belong to at least one physical group

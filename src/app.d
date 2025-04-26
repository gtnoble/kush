module app;

import std.stdio;
import std.math : abs, PI;
import core.material_point;
import core.material_body;
import core.simulation;
import models.bond_based_point;
import math.vector;

void main() {
    // Material properties (aluminum-like)
    double density = 2700.0;            // kg/m^3
    double youngsModulus = 70e9;        // Pa
    double criticalStretch = 0.05;      // 5% critical stretch

    // Discretization parameters
    double dx = 0.001;                  // Grid spacing (m)
    double horizon = 3.0 * dx;          // Horizon radius (3x grid spacing)
    double thickness = dx;              // Plate thickness (m)
    double area = dx * thickness;       // Area per point (m^2)
    double volume = area * dx;          // Volume per point (m^3)
    double mass = density * volume;     // Mass per point (kg)

    // Bond stiffness (derived from elastic properties for 2D)
    double bondStiffness = 9.0 * youngsModulus / (PI * horizon * horizon * thickness);

    // Create points in a grid (50x25 points)
    BondBasedPoint!Vector2D[] points;
    int nx = 50;
    int ny = 25;
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Skip points to create a pre-crack in the middle
            if (j == ny/2 && i < nx/2) continue;
            
            Vector2D pos = Vector2D(i * dx, j * dx);
            auto point = new BondBasedPoint!Vector2D(
                pos, mass, bondStiffness, criticalStretch
            );
            points ~= point;
        }
    }

    // Create material body
    auto body = new MaterialBody!(BondBasedPoint!Vector2D, Vector2D)(points, horizon);

    // Apply displacement boundary conditions
    for (size_t i = 0; i < body.numPoints; i++) {
        auto point = body[i];
        auto y = point.position[1];
        
        // Fix left edge, displace right edge
        if (abs(point.position[0]) < dx/2) {
            // Left edge: fixed
            point.velocity = Vector2D(0.0, 0.0);
        }
        else if (abs(point.position[0] - (nx-1)*dx) < dx/2) {
            // Right edge: constant velocity upward
            if (y < ny*dx/2) {
                point.velocity = Vector2D(0.0, -0.1);  // Bottom half moves down
            } else {
                point.velocity = Vector2D(0.0, 0.1);   // Top half moves up
            }
        }
    }

    // Simulation parameters
    double timeStep = 1e-7;  // Small time step for stability
    size_t numSteps = 1000;  // Run for 1000 steps

    writeln("Starting simulation...");
    writeln("Number of points: ", body.numPoints);
    
    // Run simulation
    simulate(body, timeStep, numSteps);
    
    writeln("Simulation complete.");

    // Output final positions (for visualization)
    writeln("\nFinal positions:");
    for (size_t i = 0; i < body.numPoints; i++) {
        auto pos = body[i].position;
        writefln("%f,%f", pos[0], pos[1]);
    }
}

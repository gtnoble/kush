module app;

import std.stdio;
import std.math : abs, PI, sqrt;
import core.material_point;
import core.material_body;
import core.simulation;
import models.bond_based_point;
import math.vector;

void main() {
    // Material properties (aluminum-like)
    double density = 2700.0;            // kg/m^3
    double youngsModulus = 70e3;        // Pa (scaled down for numerical stability)
    //double youngsModulus = 1;        // Pa (scaled down for numerical stability)
    double criticalStretch = 10;      // 5% critical stretch
    //double criticalStretch = 0.05;      // 5% critical stretch

    // Discretization parameters
    double dx = 0.001;                  // Grid spacing (m)
    double horizon = 3.0 * dx;          // Horizon radius (3x grid spacing)
    double thickness = dx;              // Plate thickness (m)
    double area = dx * thickness;       // Area per point (m^2)
    double volume = area * dx;          // Volume per point (m^3)
    double mass = density * volume;     // Mass per point (kg)

    // Bond stiffness (derived from elastic properties for 2D)
    double bondStiffness = 6.0 * youngsModulus / (PI * horizon * horizon * horizon);
    
    // Artificial viscosity parameters
    double soundSpeed = sqrt(youngsModulus / density);  // Material sound speed
    //double viscosityLinear = 1.0;   // Linear viscosity coefficient
    double viscosityLinear = 0.0;   // Linear viscosity coefficient
    //double viscosityQuad = 1.0;     // Quadratic viscosity coefficient
    double viscosityQuad = 0.0;     // Quadratic viscosity coefficient
    double dampingTimeConstant = 1e-9;  // seconds

    // Create points in a grid (50x25 points)
    BondBasedPoint!Vector2D[] points;
    int nx = 50;
    int ny = 25;
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Skip points to create a wider pre-crack in the middle
            if (abs(j - ny/2) <= 2 && i < nx/2) continue;
            
            Vector2D pos = Vector2D(i * dx, j * dx);
            BondBasedPoint!Vector2D point;

            // Calculate distances
            double distanceFromUpperCrack = abs(j - (ny/2 + 3)) * dx;
            double distanceFromRightEdge = abs(pos[0] - (nx-1)*dx);
            
            // Check conditions
            bool isNearUpperCrack = (distanceFromUpperCrack <= horizon && i < nx/2);
            bool isNearRightEdge = (distanceFromRightEdge <= horizon);
            
            // Create point based on conditions
            if (isNearUpperCrack) {
                // Points near upper crack edge: apply ramped upward force
                point = new BondBasedPoint!Vector2D(
                    pos,                     // reference position
                    mass,                    // point mass
                    bondStiffness,           // bond stiffness
                    criticalStretch,         // critical stretch
                    dampingTimeConstant,     // damping time constant
                    Vector2D(0.0, 0.0),      // zero initial velocity
                    false,                   // not fixed
                    Vector2D(0.0, 100e-3),     // target force
                    100e-9,                  // ramp duration (100ns)
                    soundSpeed,              // material sound speed
                    viscosityLinear,         // linear viscosity term
                    viscosityQuad            // quadratic viscosity term
                );
            }
            else if (isNearRightEdge) {
                // Points near right edge: fix with zero velocity
                point = new BondBasedPoint!Vector2D(
                    pos,                    // reference position
                    mass,                   // point mass
                    bondStiffness,          // bond stiffness
                    criticalStretch,        // critical stretch
                    dampingTimeConstant,    // damping time constant
                    Vector2D(0.0, 0.0),     // zero velocity
                    true,                   // fixed
                    Vector2D(0.0, 0.0),     // no external force
                    1e-6,                   // default ramp duration
                    soundSpeed,             // material sound speed
                    viscosityLinear,        // linear viscosity term
                    viscosityQuad           // quadratic viscosity term
                );
            }
            else {
                // All other points: free to move
                point = new BondBasedPoint!Vector2D(
                    pos,                    // reference position
                    mass,                   // point mass
                    bondStiffness,          // bond stiffness
                    criticalStretch,        // critical stretch
                    dampingTimeConstant,    // damping time constant
                    Vector2D(0.0, 0.0),     // zero initial velocity
                    false,                  // not fixed
                    Vector2D(0.0, 0.0),     // no external force
                    1e-6,                   // default ramp duration
                    soundSpeed,             // material sound speed
                    viscosityLinear,        // linear viscosity term
                    viscosityQuad           // quadratic viscosity term
                );
            }
            points ~= point;
        }
    }

    // Create material body
    auto body = new MaterialBody!(BondBasedPoint!Vector2D, Vector2D)(points, horizon);

    // Simulation parameters
    // Maximum time step based on stability criterion: dt < 2/sqrt(k/m)
    // where k ≈ 2e7 N/m³ and m ≈ 2.7e-9 kg
    double maxTimeStep = 1e-10;  // Maximum allowed time step
    double totalTime = maxTimeStep * 100;  // Total simulation time
    double safetyFactor = 0.1;  // Conservative safety factor for stability
    
    writeln("Starting simulation...");
    writeln("Number of points: ", body.numPoints);
    
    // Create adaptive time step strategy
    auto timeStepStrategy = new AdaptiveTimeStep!(BondBasedPoint!Vector2D, Vector2D)(
        maxTimeStep,    // Maximum allowed time step
        horizon,        // Peridynamic horizon
        safetyFactor,  // Safety factor for stability
        0.1            // Maximum allowed relative motion (10% of horizon)
    );
    
    // Run simulation with adaptive time stepping
    simulate(body, timeStepStrategy, totalTime);
    
    writeln("Simulation complete.");

    // Export results to CSV file
    body.exportToCSV("simulation_result.csv");
    writeln("Results exported to simulation_result.csv");
}

module app;

import std.stdio;
import std.math : abs, PI, sqrt;
import std.random : Random, uniform;
import core.material_point;
import core.material_body;
import core.simulation;
import models.bond_based_point;
import math.vector;

void main() {
    // Set random seed for reproducibility
    auto rnd = Random(42); // Seed value can be changed as needed
    // Material properties (aluminum-like)
    double density = 2700.0;            // kg/m^3
    //double youngsModulus = 70e3;        // Pa (scaled down for numerical stability)
    double youngsModulus = 70e9;        // Pa (scaled down for numerical stability)
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
    
    // Damping parameters
    double massTimeConstant = 1e-10;    // Mass damping time constant (seconds)
    double viscosityTimeConstant = double.infinity;   // Viscosity damping time constant (seconds)

    // Create single damper instance for all points
    auto damper = new StandardDamper!Vector2D(
        horizon,              // Neighborhood radius
        massTimeConstant,     // Mass damping time constant
        viscosityTimeConstant // Viscosity time constant
        // Using default cubicSplineInfluence function
    );

    // Simulation parameters
    // Maximum time step based on stability criterion: dt < 2/sqrt(k/m)
    // where k ≈ 2e7 N/m³ and m ≈ 2.7e-9 kg
    double initialTimeStep = 1e-20;  // Maximum allowed time step
    double maxTimeStep = 1e-9;    // Maximum time step for simulation
    double totalTime = maxTimeStep * 1000;  // Total simulation time
    double safetyFactor = 0.1;  // Conservative safety factor for stability

    // Calculate characteristic velocity as sqrt(E/ρ)
    double characteristicVelocity = sqrt(youngsModulus / density);

    // Create adaptive time step strategy with physical parameters
    auto timeStepStrategy = new AdaptiveTimeStep!(BondBasedPoint!Vector2D, Vector2D)(
        initialTimeStep,        // initialTimeStep [time]
        maxTimeStep,           // absoluteMaxTimeStep [time]
        horizon,              // horizon [length]
        characteristicVelocity * 0.01, // characteristicVelocity [length/time] (1% of material wave speed)
        100000,               // responseScaling (response time is 5x the current time step)
        safetyFactor,        // safetyFactor []
        0.1                  // maxRelativeMotion [] (10% of horizon)
    );

    
    // Sampling method enum
    enum SamplingMethod {
        Random,
        Jitter
    }

    // Create points using jitter sampling
    BondBasedPoint!Vector2D[] points;
    int nx = 50;
    int ny = 25;
    double domainWidth = nx * dx;
    double domainHeight = ny * dx;
    auto samplingMethod = SamplingMethod.Jitter;

    // Function to create a point at given position
    BondBasedPoint!Vector2D createPoint(Vector2D pos, bool isNearUpperCrack, bool isNearRightEdge) {
        if (isNearUpperCrack) {
            // Points near upper crack edge: apply ramped upward force
            return new BondBasedPoint!Vector2D(
                pos,                     // reference position
                mass,                    // point mass
                bondStiffness,          // bond stiffness
                criticalStretch,        // critical stretch
                damper,                 // damping strategy
                Vector2D(0.0, 0.0),     // zero initial velocity
                false,                  // not fixed
                Vector2D(0.0, 100e-3),  // target force
                100e-9                  // ramp duration (100ns)
            );
        }
        else if (isNearRightEdge) {
            // Points near right edge: fix with zero velocity
            return new BondBasedPoint!Vector2D(
                pos,                    // reference position
                mass,                   // point mass
                bondStiffness,         // bond stiffness
                criticalStretch,       // critical stretch
                damper,                // damping strategy
                Vector2D(0.0, 0.0),    // zero velocity
                true,                  // fixed
                Vector2D(0.0, 0.0),    // no external force
                1e-6                   // default ramp duration
            );
        }
        else {
            // All other points: free to move
            return new BondBasedPoint!Vector2D(
                pos,                    // reference position
                mass,                   // point mass
                bondStiffness,         // bond stiffness
                criticalStretch,       // critical stretch
                damper,                // damping strategy
                Vector2D(0.0, 0.0),    // zero initial velocity
                false,                 // not fixed
                Vector2D(0.0, 0.0),    // no external force
                1e-6                   // default ramp duration
            );
        }
    }

    if (samplingMethod == SamplingMethod.Jitter) {
        // Create points using jitter sampling
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                // Skip points in the crack region
                if (abs(j - ny/2) <= 2 && i < nx/2) continue;
                
                // Base position at cell corner
                double baseX = i * dx;
                double baseY = j * dx;
                
                // Add random offset within the cell
                double x = baseX + uniform(0.0, dx, rnd);
                double y = baseY + uniform(0.0, dx, rnd);
                Vector2D pos = Vector2D(x, y);
                
                // Calculate distances for point type determination
                double distanceFromUpperCrack = abs(j - (ny/2 + 3));
                double distanceFromRightEdge = abs(x - domainWidth);
                
                // Check conditions
                bool isNearUpperCrack = (distanceFromUpperCrack * dx <= horizon && i < nx/2);
                bool isNearRightEdge = (distanceFromRightEdge <= horizon);
                
                points ~= createPoint(pos, isNearUpperCrack, isNearRightEdge);
            }
        }
    } else {
        // Calculate approximate number of points (same as grid minus crack area)
        int targetPoints = cast(int)(nx * ny - (nx/2 * 5)); // 5 is crack height
        
        for (int i = 0; i < targetPoints; i++) {
            // Generate random position
            double x = uniform(0.0, domainWidth, rnd);
            double y = uniform(0.0, domainHeight, rnd);
            Vector2D pos = Vector2D(x, y);
            
            // Skip points in the crack region
            if (abs(y/dx - ny/2) <= 2 && x < domainWidth/2) continue;
            
            // Calculate distances for point type determination
            double distanceFromUpperCrack = abs(y/dx - (ny/2 + 3));
            double distanceFromRightEdge = abs(x - domainWidth);
            
            // Check conditions
            bool isNearUpperCrack = (distanceFromUpperCrack * dx <= horizon && x < domainWidth/2);
            bool isNearRightEdge = (distanceFromRightEdge <= horizon);
            
            points ~= createPoint(pos, isNearUpperCrack, isNearRightEdge);
        }
    }

    // Create material body
    auto body = new MaterialBody!(BondBasedPoint!Vector2D, Vector2D)(points, horizon);

    writeln("Starting simulation...");
    writeln("Number of points: ", body.numPoints);

    // Run simulation with adaptive time stepping
    simulate(body, timeStepStrategy, totalTime);

    writeln("Simulation complete.");

    // Export results to CSV file
    body.exportToCSV("simulation_result.csv");
    writeln("Results exported to simulation_result.csv");
}

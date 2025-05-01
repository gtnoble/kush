module io.cli;

// Standard imports
import std.getopt;
import std.stdio;
import std.conv : to;
import std.exception : enforce;
import std.format : format;

// Math imports
import math.vector;

// Core imports
import core.integration : GradientDescentSolver, GradientUpdateMode, LagrangianIntegrator;
import core.damper : StandardDamper;
import core.material_body : MaterialBody;
import core.simulation : simulate, AdaptiveTimeStep;

// Model imports
import models.bond_based_point : BondBasedPoint;

// IO imports
import io.point_loader;
import io.material_loader;
import io.simulation_loader;

/// Parsed command line arguments
struct CLIOptions {
    int dimension;           // Simulation dimension (1, 2, or 3)
    string points_file;      // Path to points configuration (JSONL)
    string materials_file;   // Path to materials configuration (JSON)
    string simulation_file;  // Path to simulation configuration (JSON)
}

/// Help text for command line options
private enum helpText = 
    "Usage: peridynamics [options]\n" ~
    "Run a peridynamics simulation using configuration files.\n\n" ~
    "Options:\n" ~
    "  -d, --dimension <dim>     Dimension of the simulation (1, 2, or 3)\n" ~
    "  -p, --points <file>       Points configuration file (JSONL)\n" ~
    "  -m, --materials <file>    Materials configuration file (JSON)\n" ~
    "  -s, --simulation <file>   Simulation configuration file (JSON)\n" ~
    "  -h, --help               Display this help message";

/// Parse command line arguments
CLIOptions parseCommandLine(string[] args) {
    CLIOptions options;
    bool help;
    
    try {
        auto helpInfo = getopt(args,
            "dimension|d", "Simulation dimension (1, 2, or 3)", &options.dimension,
            "points|p", "Points configuration file (JSONL)", &options.points_file,
            "materials|m", "Materials configuration file (JSON)", &options.materials_file,
            "simulation|s", "Simulation configuration file (JSON)", &options.simulation_file,
            "help|h", "Display this help message", &help
        );
        
        if (help) {
            writeln(helpText);
            import core.stdc.stdlib : exit;
            exit(0);
        }
        
        // Validate required arguments
        enforce(options.dimension > 0 && options.dimension <= 3,
            "Dimension must be 1, 2, or 3");
        enforce(options.points_file,
            "Points file (-p, --points) is required");
        enforce(options.materials_file,
            "Materials file (-m, --materials) is required");
        enforce(options.simulation_file,
            "Simulation file (-s, --simulation) is required");
            
    } catch (Exception e) {
        stderr.writeln("Error: ", e.msg);
        stderr.writeln(helpText);
        import core.stdc.stdlib : exit;
        exit(1);
    }
    
    return options;
}

/// Handle simulation setup and execution
struct SimulationRunner {
    CLIOptions options;

    /// Run simulation with appropriate vector type based on dimension
    void run() {
        final switch (options.dimension) {
            case 1:
                runDimensional!Vector1D();
                break;
            case 2:
                runDimensional!Vector2D();
                break;
            case 3:
                runDimensional!Vector3D();
                break;
        }
    }

private:
    /// Run simulation with specific vector type
    void runDimensional(V)() {
        
        writefln("Running %dD simulation...", V.dimension);
        writefln("- Points: %s", options.points_file);
        writefln("- Materials: %s", options.materials_file);
        writefln("- Simulation: %s", options.simulation_file);
        
        // Load configurations
        auto materials = loadMaterialConfigs!V(options.materials_file);
        auto simConfig = loadSimulationConfig(options.simulation_file);
        auto pointConfigs = loadPointConfigs!V(options.points_file);
        
        // Create points
        BondBasedPoint!V[] bondedPoints;
        foreach (pointConfig; pointConfigs) {
            // Get material properties
            auto material = materials.getMaterial(pointConfig.material);
            
            // Create damper
            auto damper = new StandardDamper!V(
                simConfig.horizon,
                simConfig.damping.mass_time_constant,
                simConfig.damping.viscosity_time_constant
            );
            
            // Combine point and material settings
            V effectiveVelocity = pointConfig.velocity;
            if (material.velocity != V.zero()) {
                effectiveVelocity = material.velocity;
            }
            
            bool effectiveFixedVelocity = pointConfig.fixed_velocity;
            if (material.fixed_velocity) {
                effectiveFixedVelocity = true;
            }
            
            // Sum point and material forces
            V totalForce = pointConfig.force + material.force;
            
            // Create point
            bondedPoints ~= new BondBasedPoint!V(
                pointConfig.position,
                calculateMass!V(material.density, pointConfig.volume),
                calculateBondStiffness!V(material.youngsModulus, simConfig.horizon),
                material.criticalStretch,
                damper,
                effectiveVelocity,
                effectiveFixedVelocity,
                totalForce,
                pointConfig.ramp_duration
            );
        }
        
        // Create material body
        auto body = new MaterialBody!(BondBasedPoint!V, V)(bondedPoints, simConfig.horizon);
        
        // Get material wave speed for time stepping
        double minWaveSpeed = materials.calculateMinWaveSpeed();
        
        // Create solver
        auto solver = new GradientDescentSolver!(BondBasedPoint!V, V)(
            simConfig.optimization.tolerance,
            simConfig.optimization.max_iterations,
            simConfig.optimization.getEffectiveStepSize(simConfig.horizon),
            simConfig.optimization.momentum,  // Move momentum before update mode
            GradientUpdateMode.StepSize
        );
        
        // Create time step strategy
        auto timeStepStrategy = new AdaptiveTimeStep!(BondBasedPoint!V, V)(
            simConfig.time_stepping.getInitialTimeStep(simConfig.horizon, minWaveSpeed),
            simConfig.time_stepping.max_step,
            simConfig.horizon,
            simConfig.time_stepping.getCharacteristicVelocity(minWaveSpeed),
            simConfig.time_stepping.response_lag,
            simConfig.time_stepping.safety_factor,
            simConfig.time_stepping.max_relative_motion
        );
        
        // Configure integrator
        auto integrator = new LagrangianIntegrator!(BondBasedPoint!V, V)(solver);

        // Run simulation
        simulate(
            body, 
            timeStepStrategy,
            integrator,
            simConfig.time_stepping.total_time,
            simConfig.output
        );
        
        writeln("Simulation complete.");
        writefln("Results written to %s", simConfig.output.csv_file);
    }

    // Calculate mass based on density and volume
    double calculateMass(V)(double density, double volume) {
        return density * volume;
    }

    // Calculate bond stiffness (derived from elastic properties)
    double calculateBondStiffness(V)(double youngsModulus, double horizon) {
        import std.math : PI;
        static if (V.dimension == 1) {
            return youngsModulus / (horizon * horizon);
        } else static if (V.dimension == 2) {
            return 6.0 * youngsModulus / (PI * horizon * horizon * horizon);
        } else {
            return 9.0 * youngsModulus / (PI * horizon * horizon * horizon * horizon);
        }
    }
}

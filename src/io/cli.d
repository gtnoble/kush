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
import core.integration : LagrangianIntegrator;
import core.damper : StandardDamper;
import core.material_body : MaterialBody;
import core.simulation : simulate, AdaptiveTimeStep;
import core.optimization : createOptimizer;

// Model imports
import models.bond_based_point : BondBasedPoint;

// IO imports
import io.point_loader;
import std.sumtype;
import io.material_loader;
import io.simulation_loader;

/// Parsed command line arguments
struct CLIOptions {
    string points_file;      // Path to points configuration (JSONL)
    string materials_file;   // Path to materials configuration (JSON)
    string simulation_file;  // Path to simulation configuration (JSON)
    string output_dir;       // Output directory for simulation results
}

/// Help text for command line options
private enum helpText = 
    "Usage: peridynamics [options]\n" ~
    "Run a peridynamics simulation using configuration files.\n\n" ~
    "Options:\n" ~
    "  -p, --points <file>       Points configuration file (JSONL)\n" ~
    "  -m, --materials <file>    Materials configuration file (JSON)\n" ~
    "  -s, --simulation <file>   Simulation configuration file (JSON)\n" ~
    "  -o, --output-dir <dir>    Output directory for simulation results\n" ~
    "  -h, --help               Display this help message";

/// Parse command line arguments
CLIOptions parseCommandLine(string[] args) {
    CLIOptions options;
    bool help;
    
    try {
        auto helpInfo = getopt(args,
            "points|p", "Points configuration file (JSONL)", &options.points_file,
            "materials|m", "Materials configuration file (JSON)", &options.materials_file,
            "simulation|s", "Simulation configuration file (JSON)", &options.simulation_file,
            "output-dir|o", "Output directory for simulation results", &options.output_dir,
            "help|h", "Display this help message", &help
        );
        
        if (help) {
            writeln(helpText);
            import core.stdc.stdlib : exit;
            exit(0);
        }
        
        // Validate required arguments
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

    /// Run simulation
    void run() {
        import io.point_loader : loadPointConfigsWithDimension, DimensionalPoints;
        
        // Get points and detect dimension
        auto loadedData = loadPointConfigsWithDimension(
            options.points_file,
            options.materials_file
        );
        
        // Match on dimension and run appropriate simulation
        loadedData.match!(
            (LoadedPoints!Vector1D p) => runWithPoints(p),
            (LoadedPoints!Vector2D p) => runWithPoints(p),
            (LoadedPoints!Vector3D p) => runWithPoints(p)
        );
    }

private:
    /// Process loaded points of any dimension
    private void runWithPoints(V)(LoadedPoints!V loadedData) {
        
        writefln("Running %dD simulation...", V.dimension);
        writefln("- Points: %s", options.points_file);
        writefln("- Materials: %s", options.materials_file);
        writefln("- Simulation: %s", options.simulation_file);
        if (options.output_dir) {
            writefln("- Output directory: %s", options.output_dir);
        }
        
        // Load simulation config
        auto simConfig = loadSimulationConfig(options.simulation_file);
        
        // Create points
        BondBasedPoint!V[] bondedPoints;
        foreach (point; loadedData.points) {
            // Get material properties by merging groups
            auto material = loadedData.materials.mergeGroups(point.material_groups);
            
            // Create damper
            auto damper = new StandardDamper!V(
                simConfig.horizon,
                simConfig.damping.mass_time_constant,
                simConfig.damping.viscosity_time_constant
            );
            
            // Get scaled force from material config
            V scaledForce = V.zero();
            foreach (group; point.material_groups) {
                if (loadedData.materials.getMaterial(group).force != V.zero()) {
                    scaledForce = loadedData.materials.getScaledForce(group);
                    break;  // Use first non-zero force found
                }
            }

            // Create point (properties come from merged material)
            bondedPoints ~= new BondBasedPoint!V(
                point.position,
                calculateMass!V(material.density, point.volume),
                calculateBondStiffness!V(material.youngsModulus, simConfig.horizon),
                material.criticalStretch,
                damper,
                material.velocity,
                material.fixed_velocity,
                scaledForce,
                material.ramp_duration
            );
        }
        
        // Create material body
        auto body = new MaterialBody!(BondBasedPoint!V, V)(bondedPoints, simConfig.horizon);
        
        // Get material wave speed for time stepping
        double minWaveSpeed = loadedData.materials.calculateMinWaveSpeed();
        
        // Create solver using core optimizer factory
        auto solver = createOptimizer!V(
            simConfig.optimization,
            simConfig.horizon
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

        // Create output directory if specified
        if (options.output_dir) {
            import std.file : exists, mkdirRecurse;
            import std.path : buildPath;
            
            if (!exists(options.output_dir)) {
                mkdirRecurse(options.output_dir);
            }
            
            // Update output paths with output directory
            simConfig.output.csv_file = buildPath(options.output_dir, simConfig.output.csv_file);
        }

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

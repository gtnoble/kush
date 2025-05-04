module io.simulation_loader;

import core.optimization : OptimizationSolver;
import core.material_point : isMaterialPoint;
import std.json;
import std.stdio;
import std.exception : enforce;
import std.format : format;
import std.file : readText;

/// Parameters for optimization
struct OptimizationConfig {
    double tolerance = 1e-6;
    int max_iterations = 10;
    string solver_type = "gradient_descent";  // "gradient_descent" or "parallel_tempering"

    // Gradient descent specific settings
    double momentum = 0.1;
    string gradient_mode = "step_size";  // "step_size" or "learning_rate"
    double learning_rate = 0.01;
    
    struct GradientStepSize {
        double value = 1e-4;
        double horizon_fraction;  // Optional: if specified, step = horizon * fraction
    }
    GradientStepSize gradient_step_size;

    size_t getNumReplicas() const {
        import std.parallelism : totalCPUs;
        if (solver_type != "parallel_tempering") return 0;
        if (parallel_tempering.num_replicas > 0) 
            return parallel_tempering.num_replicas;
        return totalCPUs;
    }

    // Parallel tempering specific settings
    struct ParallelTempering {
        size_t num_replicas;  // Defaults to totalCPUs if not specified
        double min_temperature = 0.1;
        double max_temperature = 2.0;
    }
    ParallelTempering parallel_tempering;
    
    /// Calculate the effective gradient step size
    double getEffectiveStepSize(double horizon) const {
        if (gradient_step_size.horizon_fraction > 0.0) {
            return horizon * gradient_step_size.horizon_fraction;
        }
        return gradient_step_size.value;
    }
}

/// Parameters for adaptive time stepping
struct TimeSteppingConfig {
    double initial_step;            // Optional: auto-calculated as (horizon/wave_speed) * 0.0001
    double max_step = 1e-4;
    double total_time = 3e-3;
    double safety_factor = 0.1;
    double characteristic_velocity; // Optional: auto-calculated as min(sqrt(E/ρ)) from materials
    double max_relative_motion = 0.1;
    double response_lag = 1000;
    
    /// Calculate initial time step if not specified
    double getInitialTimeStep(double horizon, double wave_speed) const {
        if (initial_step > 0.0) {
            return initial_step;
        }
        return (horizon / wave_speed) * 0.0001;
    }
    
    /// Get characteristic velocity (use provided or material-based)
    double getCharacteristicVelocity(double material_wave_speed) const {
        if (characteristic_velocity > 0.0) {
            return characteristic_velocity;
        }
        return material_wave_speed * 0.01;
    }
}

/// Parameters for damping
struct DampingConfig {
    double mass_time_constant = 1e-4;
    double viscosity_time_constant;  // Optional: omitted means no viscosity damping
    
    bool hasViscosityDamping() const {
        return viscosity_time_constant > 0.0;
    }
}

/// Parameters for simulation output
struct OutputConfig {
    string csv_file = "simulation_result.csv";
    int step_interval;        // Save every N steps (mutually exclusive with time_interval)
    double time_interval;     // Save every T seconds (mutually exclusive with step_interval)
}

/// Complete simulation configuration
struct SimulationConfig {
    double horizon;
    OptimizationConfig optimization;
    TimeSteppingConfig time_stepping;
    DampingConfig damping;
    OutputConfig output;
}

/// Parse optimization configuration from JSON
private OptimizationConfig parseOptimizationConfig(JSONValue json) {
    OptimizationConfig config;
    
    // Parse solver type
    if ("solver_type" in json) {
        config.solver_type = json["solver_type"].get!string;
        enforce(config.solver_type == "gradient_descent" || 
               config.solver_type == "parallel_tempering",
            "Solver type must be 'gradient_descent' or 'parallel_tempering'");
    }
    
    // Parse tolerance
    if ("tolerance" in json) {
        config.tolerance = json["tolerance"].get!double;
        enforce(config.tolerance > 0.0, "Tolerance must be positive");
    }
    
    // Parse max iterations
    if ("max_iterations" in json) {
        config.max_iterations = json["max_iterations"].get!int;
        enforce(config.max_iterations > 0, "Max iterations must be positive");
    }
    
    // Parse momentum
    if ("momentum" in json) {
        config.momentum = json["momentum"].get!double;
        enforce(config.momentum >= 0.0 && config.momentum < 1.0, 
            "Momentum must be in range [0, 1)");
    }
    
    if ("parallel_tempering" in json) {
        auto pt = json["parallel_tempering"];
        
        if ("num_replicas" in pt) {
            config.parallel_tempering.num_replicas = pt["num_replicas"].get!size_t;
            enforce(config.parallel_tempering.num_replicas > 1,
                "Number of replicas must be greater than 1");
        }
        
        if ("min_temperature" in pt) {
            config.parallel_tempering.min_temperature = pt["min_temperature"].get!double;
            enforce(config.parallel_tempering.min_temperature > 0.0,
                "Minimum temperature must be positive");
        }
        
        if ("max_temperature" in pt) {
            config.parallel_tempering.max_temperature = pt["max_temperature"].get!double;
            enforce(config.parallel_tempering.max_temperature > 
                   config.parallel_tempering.min_temperature,
                "Maximum temperature must be greater than minimum temperature");
        }
    }

    // Parse learning rate
    if ("learning_rate" in json) {
        config.learning_rate = json["learning_rate"].get!double;
        enforce(config.learning_rate > 0.0, "Learning rate must be positive");
    }

    // Parse gradient mode
    if ("gradient_mode" in json) {
        config.gradient_mode = json["gradient_mode"].get!string;
        enforce(config.gradient_mode == "step_size" || config.gradient_mode == "learning_rate",
            "Gradient mode must be either 'step_size' or 'learning_rate'");
    }
    
    if ("gradient_step_size" in json) {
        auto step = json["gradient_step_size"];
        if ("value" in step) {
            config.gradient_step_size.value = step["value"].get!double;
            enforce(config.gradient_step_size.value > 0.0, 
                "Gradient step size value must be positive");
        }
        if ("horizon_fraction" in step) {
            config.gradient_step_size.horizon_fraction = 
                step["horizon_fraction"].get!double;
            enforce(config.gradient_step_size.horizon_fraction > 0.0, 
                "Horizon fraction must be positive");
        }
    }
    
    return config;
}

// Create an optimizer from the configuration
OptimizationSolver!(T, V) createOptimizer(T, V)(
    const OptimizationConfig config, double horizon
) if (isMaterialPoint!(T, V)) {
    import core.optimization : createOptimizer;

    return .createOptimizer!(T, V)(
        config.tolerance,
        config.max_iterations,
        config.solver_type,
        config.learning_rate,
        config.getEffectiveStepSize(horizon),
        config.momentum,
        config.gradient_mode,
        config.getNumReplicas(),
        config.parallel_tempering.min_temperature,
        config.parallel_tempering.max_temperature
    );
}
/// Parse time stepping configuration from JSON
private TimeSteppingConfig parseTimeSteppingConfig(JSONValue json) {
    TimeSteppingConfig config;
    
    // Parse initial step (optional)
    if ("initial_step" in json) {
        config.initial_step = json["initial_step"].get!double;
        enforce(config.initial_step > 0.0, "Initial time step must be positive");
    }
    
    // Parse max step
    if ("max_step" in json) {
        config.max_step = json["max_step"].get!double;
        enforce(config.max_step > 0.0, "Max time step must be positive");
    }
    
    // Parse total time
    if ("total_time" in json) {
        config.total_time = json["total_time"].get!double;
        enforce(config.total_time > 0.0, "Total time must be positive");
    }
    
    // Parse safety factor
    if ("safety_factor" in json) {
        config.safety_factor = json["safety_factor"].get!double;
        enforce(config.safety_factor > 0.0 && config.safety_factor <= 1.0,
            "Safety factor must be in range (0, 1]");
    }
    
    // Parse characteristic velocity (optional)
    if ("characteristic_velocity" in json) {
        config.characteristic_velocity = json["characteristic_velocity"].get!double;
        enforce(config.characteristic_velocity > 0.0, 
            "Characteristic velocity must be positive");
    }
    
    // Parse max relative motion
    if ("max_relative_motion" in json) {
        config.max_relative_motion = json["max_relative_motion"].get!double;
        enforce(config.max_relative_motion > 0.0, 
            "Max relative motion must be positive");
    }
    
    // Parse response lag
    if ("response_lag" in json) {
        config.response_lag = json["response_lag"].get!double;
        enforce(config.response_lag > 0.0, "Response lag must be positive");
    }
    
    return config;
}

/// Parse damping configuration from JSON
private DampingConfig parseDampingConfig(JSONValue json) {
    DampingConfig config;
    
    // Parse mass time constant
    if ("mass_time_constant" in json) {
        config.mass_time_constant = json["mass_time_constant"].get!double;
        enforce(config.mass_time_constant > 0.0, 
            "Mass time constant must be positive");
    }
    
    // Parse viscosity time constant (optional)
    if ("viscosity_time_constant" in json) {
        config.viscosity_time_constant = 
            json["viscosity_time_constant"].get!double;
        enforce(config.viscosity_time_constant > 0.0, 
            "Viscosity time constant must be positive");
    }
    
    return config;
}

/// Parse output configuration from JSON
private OutputConfig parseOutputConfig(JSONValue json) {
    OutputConfig config;
    
    // Parse CSV file path
    if ("csv_file" in json) {
        config.csv_file = json["csv_file"].get!string;
    }
    
    // Handle output intervals (mutually exclusive)
    bool hasStepInterval = ("step_interval" in json) !is null;
    bool hasTimeInterval = ("time_interval" in json) !is null;
    
    enforce(hasStepInterval || hasTimeInterval, 
        "Either step_interval or time_interval must be specified");
    enforce(!(hasStepInterval && hasTimeInterval), 
        "Cannot specify both step_interval and time_interval");
    
    if (hasStepInterval) {
        config.step_interval = json["step_interval"].get!int;
        enforce(config.step_interval > 0, "Step interval must be positive");
    }
    
    if (hasTimeInterval) {
        config.time_interval = json["time_interval"].get!double;
        enforce(config.time_interval > 0.0, "Time interval must be positive");
    }
    
    return config;
}

/// Load simulation configuration from a JSON file
SimulationConfig loadSimulationConfig(string filepath) {
    import std.file : exists;
    
    enforce(exists(filepath), "Simulation file does not exist: " ~ filepath);
    
    // Read and parse JSON file
    string jsonText = readText(filepath);
    JSONValue json = parseJSON(jsonText);
    
    SimulationConfig config;
    
    // Parse horizon (required)
    enforce("horizon" in json, "Missing required field: horizon");
    config.horizon = json["horizon"].get!double;
    enforce(config.horizon > 0.0, "Horizon must be positive");
    
    // Parse optimization settings
    if ("optimization" in json) {
        config.optimization = parseOptimizationConfig(json["optimization"]);
    }
    
    // Parse time stepping settings
    if ("time_stepping" in json) {
        config.time_stepping = parseTimeSteppingConfig(json["time_stepping"]);
    }
    
    // Parse damping settings
    if ("damping" in json) {
        config.damping = parseDampingConfig(json["damping"]);
    }
    
    // Parse output settings
    if ("output" in json) {
        config.output = parseOutputConfig(json["output"]);
    }
    
    return config;
}

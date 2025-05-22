module io.simulation_loader;

import core.optimization : OptimizationSolver;
import core.material_point : isMaterialPoint;
import std.json;
import std.stdio;
import std.exception : enforce;
import std.format : format;
import std.file : readText;

/// Step size configuration with optional horizon-based scaling
struct StepSize {
    double value = 1e-4;
    double horizon_fraction;  // Optional: if specified, step = horizon * fraction

    double getEffectiveValue(double horizon) const {
        if (horizon_fraction > 0.0) {
            return horizon * horizon_fraction;
        }
        return value;
    }
}

/// Parameters for gradient descent optimization
struct GradientDescent {
    double momentum = 0.9;
    string gradient_mode = "step_size";  // "step_size", "learning_rate", "bb1", "bb2", or "bb-auto"
    double learning_rate = 0.01;         // Used in learning_rate mode
    
    /// Configuration for finite difference calculations
    struct FiniteDifferenceConfig {
        int order = 2;               // Order of accuracy (2, 4, 6, or 8)
        double step_size = 1e-6;     // Step size for finite differences
    }
    FiniteDifferenceConfig finite_difference;
    
    StepSize initial_step;     // Initial step size for all modes
    StepSize min_step;         // Minimum allowed step size
    StepSize max_step;         // Maximum allowed step size
    StepSize gradient_step_size;  // Step size for gradient mode

    /// Get effective initial step size
    double getEffectiveInitialStep(double horizon) const {
        return initial_step.getEffectiveValue(horizon);
    }
    
    /// Get effective minimum step size
    double getEffectiveMinStep(double horizon) const {
        return min_step.getEffectiveValue(horizon);
    }
    
    /// Get effective maximum step size
    double getEffectiveMaxStep(double horizon) const {
        return max_step.getEffectiveValue(horizon);
    }

    /// Get effective gradient step size
    double getEffectiveGradientStep(double horizon) const {
        return gradient_step_size.getEffectiveValue(horizon);
    }

    /// Check if mode is Barzilai-Borwein variant
    bool isBarzilaiborweinMode() const {
        return gradient_mode == "bb1" || gradient_mode == "bb2" || gradient_mode == "bb-auto";
    }
}

/// Parameters for optimization
struct OptimizationConfig {
    double tolerance = 1e-6;
    int max_iterations = 10;
    string solver_type = "gradient_descent";  // "gradient_descent", "parallel_tempering", or "lbfgs"
    GradientDescent gradient_descent;
    
    /// L-BFGS specific settings
    struct LBFGSConfig {
        size_t memory_size = 10;     // Number of past updates to store
        StepSize initial_step;       // Initial Hessian scaling
    }
    LBFGSConfig lbfgs;

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
        size_t num_processes;  // Defaults to totalCPUs if not specified
        double min_temperature = 0.1;
        double max_temperature = 2.0;

        struct ProposalStepSize {
            double value = 1e-4;
            double horizon_fraction;  // Optional: if specified, step = horizon * fraction
        }
        ProposalStepSize proposal_step_size;
    }

    size_t getNumProcesses() const {
        import std.parallelism : totalCPUs;
        if (solver_type != "parallel_tempering") return 0;
        if (parallel_tempering.num_processes > 0)
            return parallel_tempering.num_processes;
        return totalCPUs;
    }
    ParallelTempering parallel_tempering;

    /// Calculate the effective proposal step size for parallel tempering
    double getEffectiveProposalStepSize(double horizon) const {
        if (parallel_tempering.proposal_step_size.horizon_fraction > 0.0) {
            return horizon * parallel_tempering.proposal_step_size.horizon_fraction;
        }
        return parallel_tempering.proposal_step_size.value;
    }
    
    /// Calculate the effective gradient step size
    double getEffectiveStepSize(double horizon) const {
        if (gradient_descent.gradient_step_size.horizon_fraction > 0.0) {
            return horizon * gradient_descent.gradient_step_size.horizon_fraction;
        }
        return gradient_descent.gradient_step_size.value;
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
               config.solver_type == "parallel_tempering" ||
               config.solver_type == "lbfgs",
            "Solver type must be 'gradient_descent', 'parallel_tempering', or 'lbfgs'");
    }
    
    // Parse tolerance
    if ("tolerance" in json) {
        config.tolerance = json["tolerance"].get!double;
        enforce(config.tolerance >= 0.0, "Tolerance must be non-negative");
    }
    
    // Parse max iterations
    if ("max_iterations" in json) {
        config.max_iterations = json["max_iterations"].get!int;
        enforce(config.max_iterations > 0, "Max iterations must be positive");
    }
    
    // Parse gradient descent settings
    if ("gradient_descent" in json || config.solver_type == "gradient_descent") {
        auto gd = "gradient_descent" in json ? json["gradient_descent"] : json;
        
        if ("momentum" in gd) {
            config.gradient_descent.momentum = gd["momentum"].get!double;
            enforce(config.gradient_descent.momentum >= 0.0 && config.gradient_descent.momentum < 1.0,
                "Momentum must be in range [0, 1)");
        }

        if ("gradient_mode" in gd) {
            config.gradient_descent.gradient_mode = gd["gradient_mode"].get!string;
            enforce(config.gradient_descent.gradient_mode == "step_size" || 
                   config.gradient_descent.gradient_mode == "learning_rate" ||
                   config.gradient_descent.gradient_mode == "bb1" ||
                   config.gradient_descent.gradient_mode == "bb2" ||
                   config.gradient_descent.gradient_mode == "bb-auto",
                "Gradient mode must be one of: step_size, learning_rate, bb1, bb2, bb-auto");
        }

        if ("learning_rate" in gd) {
            config.gradient_descent.learning_rate = gd["learning_rate"].get!double;
            enforce(config.gradient_descent.learning_rate > 0.0, 
                "Learning rate must be positive");
        }

        if ("finite_difference" in gd) {
            auto fd = gd["finite_difference"];
            if ("order" in fd) {
                config.gradient_descent.finite_difference.order = fd["order"].get!int;
                enforce(config.gradient_descent.finite_difference.order >= 2 && 
                       config.gradient_descent.finite_difference.order % 2 == 0 &&
                       config.gradient_descent.finite_difference.order <= 8,
                    "Finite difference order must be 2, 4, 6, or 8");
            }
            if ("step_size" in fd) {
                config.gradient_descent.finite_difference.step_size = fd["step_size"].get!double;
                enforce(config.gradient_descent.finite_difference.step_size > 0.0,
                    "Finite difference step size must be positive");
            }
        }

        // Parse step sizes
        void parseStepSize(string field, ref StepSize stepSize) {
            if (field in gd) {
                auto step = gd[field];
                if ("value" in step) {
                    stepSize.value = step["value"].get!double;
                    //enforce(stepSize.value > 0.0, field ~ " value must be positive");
                }
                if ("horizon_fraction" in step) {
                    stepSize.horizon_fraction = step["horizon_fraction"].get!double;
                    enforce(stepSize.horizon_fraction > 0.0, 
                        field ~ " horizon fraction must be positive");
                }
            }
        }

        parseStepSize("initial_step", config.gradient_descent.initial_step);
        parseStepSize("min_step", config.gradient_descent.min_step);
        parseStepSize("max_step", config.gradient_descent.max_step);
        parseStepSize("gradient_step_size", config.gradient_descent.gradient_step_size);
    }

    // Parse L-BFGS settings
    if ("lbfgs" in json) {
        auto lbfgs = json["lbfgs"];
        
        if ("memory_size" in lbfgs) {
            config.lbfgs.memory_size = lbfgs["memory_size"].get!size_t;
            enforce(config.lbfgs.memory_size > 0,
                "L-BFGS memory size must be positive");
        }
        
        // Parse initial step size
        if ("initial_step" in lbfgs) {
            auto step = lbfgs["initial_step"];
            if ("value" in step) {
                config.lbfgs.initial_step.value = step["value"].get!double;
                enforce(config.lbfgs.initial_step.value > 0.0,
                    "Initial step size value must be positive");
            }
            if ("horizon_fraction" in step) {
                config.lbfgs.initial_step.horizon_fraction = step["horizon_fraction"].get!double;
                enforce(config.lbfgs.initial_step.horizon_fraction > 0.0,
                    "Horizon fraction must be positive");
            }
        }
    }

    if ("parallel_tempering" in json) {
        auto pt = json["parallel_tempering"];
        
        if ("num_replicas" in pt) {
            config.parallel_tempering.num_replicas = pt["num_replicas"].get!size_t;
            enforce(config.parallel_tempering.num_replicas > 1,
                "Number of replicas must be greater than 1");
        }

        if ("num_processes" in pt) {
            config.parallel_tempering.num_processes = pt["num_processes"].get!size_t;
            enforce(config.parallel_tempering.num_processes > 0,
                "Number of processes must be positive");
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

        if ("proposal_step_size" in pt) {
            auto step = pt["proposal_step_size"];
            if ("value" in step) {
                config.parallel_tempering.proposal_step_size.value = step["value"].get!double;
                enforce(config.parallel_tempering.proposal_step_size.value > 0.0,
                    "Proposal step size value must be positive");
            }
            if ("horizon_fraction" in step) {
                config.parallel_tempering.proposal_step_size.horizon_fraction = 
                    step["horizon_fraction"].get!double;
                enforce(config.parallel_tempering.proposal_step_size.horizon_fraction > 0.0,
                    "Horizon fraction must be positive");
            }
        }
    }

    
    return config;
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

module core.optimization;

import math.vector;
import std.math : abs, sqrt, exp, cos;
import std.math.traits : isNaN;
import std.algorithm : min, max, startsWith;
import std.random : uniform01;
import std.mathspecial : normalDistributionInverse;
import std.stdio;
import core.sys.posix.unistd : fork, pid_t;
import core.sys.posix.signal : 
    kill, SIGTERM;
import core.sys.posix.sys.wait : wait;
import core.stdc.stdlib : exit;
import std.format : format;
import std.parallelism : totalCPUs;
import std.process : thisProcessID;
import std.uuid : randomUUID;
import std.exception : enforce;
import std.conv : to;
import jumbomessage;
import io.simulation_loader : OptimizationConfig;

// Helper function to calculate BB step size
private double calculateBBStepSize(
    const DynamicVector!double gradient,
    const DynamicVector!double state,
    const DynamicVector!double previousGradient,
    const DynamicVector!double previousState,
    GradientUpdateMode updateMode
) {
    auto s = state - previousState;          // Change in position
    auto y = gradient - previousGradient;    // Change in gradient
    
    double s_dot_y = s.dot(y);  // s^T y
    double y_dot_y = y.dot(y);  // y^T y
    double s_dot_s = s.dot(s);  // s^T s
    
    double bb1 = s_dot_y / y_dot_y;  // BB1 step size
    double bb2 = s_dot_s / s_dot_y;  // BB2 step size
    
    if (isNaN(bb1) && isNaN(bb2)) {
        return 0;
    }
    else if (isNaN(bb1)) {
        return bb2;
    }
    else if (isNaN(bb2)) {
        return bb1;
    }
    
    // Auto mode selection based on which step gives better descent direction
    if (updateMode == GradientUpdateMode.BBAuto) {
        // Choose BB2 if oscillating (indicated by negative s_dot_y)
        return (s_dot_y < 0) ? bb2 : bb1;
    }
    else if (updateMode == GradientUpdateMode.BB1) {
        return bb1;
    }
    else if (updateMode == GradientUpdateMode.BB2) {
        return bb2;
    }
    else {
        assert(0, "Invalid update mode");
    }
}

// Helper for normal distribution sampling
private double normal(double mean, double stddev) {
    return mean + stddev * normalDistributionInverse(uniform01());
}

// Test ProcessManager
unittest {
    auto manager = new ProcessManager(2);
    assert(manager._pids.length == 2);  // Check array initialization
}

// Test GradientMessage serialization
unittest {
    // Create a test message
    auto msg = GradientMessage(1, 2, 3.14);

    // Convert to bytes
    auto bytes = msg.toBytes();

    // Convert back and verify
    auto recovered = GradientMessage.fromBytes(bytes);
    assert(recovered.workerId == 1, "WorkerId not preserved in serialization");
    assert(recovered.index == 2, "Index not preserved in serialization");
    assert(abs(recovered.value - 3.14) < 1e-10, "Value not preserved in serialization");
}

// Test WorkerInput and WorkerResult
unittest {
    // Test WorkerInput serialization
    {
        auto state = DynamicVector!double([1.0, 2.0, 3.0]);
        auto input = WorkerInput!(DynamicVector!double)(state, 10.5, 0.5);  // state, energy, temperature
        
        // Convert to bytes
        auto bytes = input.toBytes();
        
        // Convert back and verify
        auto recovered = WorkerInput!(DynamicVector!double).fromBytes(bytes);
        assert(recovered.state.length == 3);
        assert(abs(recovered.state[0] - 1.0) < 1e-10);
        assert(abs(recovered.state[1] - 2.0) < 1e-10);
        assert(abs(recovered.state[2] - 3.0) < 1e-10);
        assert(abs(recovered.energy - 10.5) < 1e-10);
        assert(abs(recovered.temperature - 0.5) < 1e-10);
    }

    // Test WorkerResult serialization
    {
        auto state = DynamicVector!double([4.0, 5.0, 6.0]);
        auto result = WorkerResult!(DynamicVector!double)(state, 20.5);  // state, energy
        
        // Convert to bytes
        auto bytes = result.toBytes();
        
        // Convert back and verify
        auto recovered = WorkerResult!(DynamicVector!double).fromBytes(bytes);
        assert(recovered.state.length == 3);
        assert(abs(recovered.state[0] - 4.0) < 1e-10);
        assert(abs(recovered.state[1] - 5.0) < 1e-10);
        assert(abs(recovered.state[2] - 6.0) < 1e-10);
        assert(abs(recovered.energy - 20.5) < 1e-10);
    }
}

// Test TemperingQueueManager
unittest {
    auto manager = new TemperingQueueManager!(DynamicVector!double)();
    
    // Test queue name generation
    assert(manager.getInputQueueName().startsWith("tempering_input_"));
    assert(manager.getResultsQueueName().startsWith("tempering_results_"));
    
    // Test queue initialization
    manager.initialize();
    assert(manager.getInputQueue() !is null);
    assert(manager.getResultsQueue() !is null);
    
    // Test cleanup
    manager.cleanup();
}

/**
 * Process manager for handling worker processes.
 * Manages spawning, monitoring, and cleanup of worker processes.
 */
class ProcessManager {
    private:
        pid_t[] _pids;
        size_t _numProcesses;
        
    public:
        this(size_t numProcesses) {
            _numProcesses = numProcesses;
            _pids = new pid_t[numProcesses];
        }
        
        void spawnWorkers(void delegate(size_t) worker) {
            foreach (i; 0.._numProcesses) {
                _pids[i] = fork();
                if (_pids[i] == 0) {  // Child process
                    try {
                        worker(i);
                    } catch (Exception e) {
                        stderr.writeln("Child process ", i, " error: ", e.msg);
                    }
                    exit(0);
                }
            }
        }
        
        void cleanupProcesses() {
            // Send SIGTERM to all workers
            foreach (pid; _pids) {
                if (pid != 0) {
                    kill(pid, SIGTERM);
                    wait(null);  // Wait for each process to exit
                }
            }
        }
}

private struct WorkerInput(V) {
    DynamicVector!double state;
    double energy;
    double temperature;

    ubyte[] toBytes() {
        ubyte[] buffer;

        // Serialize current state
        buffer ~= state.toBytes();

        // Serialize energy and temperature
        buffer ~= cast(ubyte[])(&energy)[0..1];
        buffer ~= cast(ubyte[])(&temperature)[0..1];

        return buffer;
    }

    static WorkerInput!V fromBytes(ubyte[] data) {
        size_t offset = 0;
        DynamicVector!double state;
        
        // Deserialize state
        offset += DynamicVector!double.fromBytes(data, state);
        
        // Read energy and temperature
        double energy = *cast(double*)(data.ptr + offset);
        offset += double.sizeof;
        double temperature = *cast(double*)(data.ptr + offset);
        
        return WorkerInput!V(state, energy, temperature);
    }

    void send(JumboMessageQueue queue) {
        queue.send(this.toBytes());
    }

    static WorkerInput!V receive(JumboMessageQueue queue) {
        return fromBytes(queue.receive());
    }
}

private struct WorkerResult(V) {
    DynamicVector!double state;
    double energy;

    ubyte[] toBytes() {
        ubyte[] buffer;
        
        // Serialize state
        buffer ~= state.toBytes();

        // Serialize energy
        buffer ~= cast(ubyte[])(&energy)[0..1];

        return buffer;
    }

    static WorkerResult!V fromBytes(ubyte[] data) {
        size_t offset = 0;
        DynamicVector!double state;
        
        // Deserialize state
        offset += DynamicVector!double.fromBytes(data, state);
        
        // Read energy
        double energy = *cast(double*)(data.ptr + offset);
        
        return WorkerResult!V(state, energy);
    }

    void send(JumboMessageQueue queue) {
        queue.send(this.toBytes());
    }

    static WorkerResult!V receive(JumboMessageQueue queue) {
        return fromBytes(queue.receive());
    }
}

// Test parallel tempering temperature management and replica handling
unittest {
    // Test temperature initialization
    {
        size_t numReplicas = 4;
        double minTemp = 0.1;
        double maxTemp = 2.0;
        
        auto temps = (new class {
            double[] initializeTemperatures() {
                import std.math : pow;
                auto temperatures = new double[numReplicas];
                double ratio = pow(maxTemp / minTemp, 1.0 / (numReplicas - 1));
                foreach (i; 0..numReplicas) {
                    temperatures[i] = minTemp * pow(ratio, i);
                }
                return temperatures;
            }
        }).initializeTemperatures();
        
        // Check temperature bounds
        assert(abs(temps[0] - minTemp) < 1e-10, "Incorrect minimum temperature");
        assert(abs(temps[$-1] - maxTemp) < 1e-10, "Incorrect maximum temperature");
        
        // Check geometric spacing
        double ratio = temps[1] / temps[0];
        foreach(i; 1..temps.length-1) {
            assert(abs(temps[i+1] / temps[i] - ratio) < 1e-10,
                "Temperatures not geometrically spaced");
        }
    }
    
    // Test temperature reordering
    {
        // Mock temperature and energy arrays
        double[] temps = [0.1, 0.3, 0.9, 2.0];  // Initial temperatures
        double[] energies = [5.0, 2.0, 8.0, 1.0];  // Energies of replicas
        
        // Create test reordering function
        auto reorder = (new class {
            double[] reorderTemperatures(double[] temperatures, double[] energies) {
                import std.algorithm : sort;
                
                struct StateEnergy {
                    size_t stateIndex;
                    double energy;
                }
                
                auto allStates = new StateEnergy[temperatures.length];
                foreach (i; 0..temperatures.length) {
                    allStates[i] = StateEnergy(i, energies[i]);
                }
                
                sort!((a, b) => a.energy < b.energy)(allStates);
                
                auto newTemps = new double[temperatures.length];
                foreach (i; 0..temperatures.length) {
                    newTemps[allStates[i].stateIndex] = temperatures[i];
                }
                
                return newTemps;
            }
        }).reorderTemperatures(temps, energies);
        
        // Check reordering - lowest energy should get lowest temperature
        assert(abs(reorder[3] - 0.1) < 1e-10,  // State with energy 1.0 gets lowest temp
            "Incorrect temperature assignment for lowest energy state");
        assert(abs(reorder[1] - 0.3) < 1e-10,  // State with energy 2.0 gets second lowest
            "Incorrect temperature assignment for second lowest energy state");
        assert(abs(reorder[0] - 0.9) < 1e-10,  // State with energy 5.0
            "Incorrect temperature assignment for third lowest energy state");
        assert(abs(reorder[2] - 2.0) < 1e-10,  // State with energy 8.0 gets highest temp
            "Incorrect temperature assignment for highest energy state");
    }
}

// Test L-BFGS optimization on standard test functions
unittest {
    // Test on simple quadratic function
    {
        // f(x) = x^T x (minimum at origin)
        auto objective = delegate (const DynamicVector!double x) => x.dot(x);
        
        // Initialize with point away from minimum
        auto x0 = DynamicVector!double([1.0, 1.0]);
        auto config = OptimizationConfig();
        config.solver_type = "lbfgs";
        config.tolerance = 1e-6;
        config.max_iterations = 100;
        config.lbfgs.memory_size = 5;
        config.lbfgs.initial_step.value = 1.0;
        
        auto optimizer = createOptimizer!(DynamicVector!double)(config, 1.0);
        auto gradient = createFiniteDifferenceGradient(objective, 2);
        
        // Optimize
        auto result = optimizer(x0, objective, gradient);
        
        // Check convergence to origin
        assert(abs(result[0]) < 1e-5 && abs(result[1]) < 1e-5,
            "L-BFGS failed to converge to minimum");
    }
    
    // Test on Rosenbrock function
    {
        // f(x,y) = (1-x)^2 + 100(y-x^2)^2 (minimum at [1,1])
        auto objective = delegate (const DynamicVector!double x) {
            double term1 = (1.0 - x[0]) * (1.0 - x[0]);
            double term2 = 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
            return term1 + term2;
        };
        
        auto x0 = DynamicVector!double([-1.0, 2.0]);
        auto config = OptimizationConfig();
        config.solver_type = "lbfgs";
        config.tolerance = 1e-6;
        config.max_iterations = 1000;
        config.lbfgs.memory_size = 10;
        config.lbfgs.initial_step.value = 0.1;
        
        auto optimizer = createOptimizer!(DynamicVector!double)(config, 1.0);
        auto gradient = createFiniteDifferenceGradient(objective, 4);
        
        // Optimize
        auto result = optimizer(x0, objective, gradient);
        
        // Check convergence to [1,1]
        assert(abs(result[0] - 1.0) < 1e-4 && abs(result[1] - 1.0) < 1e-4,
            "L-BFGS failed to find Rosenbrock minimum");
    }
}

// Test CircularBuffer and createOptimizer
unittest {
    // Test CircularBuffer
    {
        auto buffer = CircularBuffer!int(3);
        
        // Test initial state
        assert(buffer.length == 0);
        assert(buffer.capacity == 3);
        
        // Test pushing elements
        buffer.push(1);
        assert(buffer.length == 1);
        assert(buffer[0] == 1);
        
        buffer.push(2);
        buffer.push(3);
        assert(buffer.length == 3);
        assert(buffer[0] == 3);  // Most recent
        assert(buffer[1] == 2);
        assert(buffer[2] == 1);  // Oldest
        
        // Test wrapping around
        buffer.push(4);
        assert(buffer.length == 3);
        assert(buffer[0] == 4);
        assert(buffer[1] == 3);
        assert(buffer[2] == 2);
        
        // Test clear
        buffer.clear();
        assert(buffer.length == 0);
        assert(buffer.capacity == 3);
    }
    
    // Test createOptimizer error handling
    {
        auto config = OptimizationConfig();
        config.solver_type = "invalid_solver";
        
        try {
            auto optimizer = createOptimizer!(DynamicVector!double)(config, 1.0);
            assert(false, "Should have thrown exception for invalid solver type");
        } catch (Exception e) {
            assert(e.msg == "Unknown solver type: invalid_solver");
        }
    }
}

/**
 * Queue manager for parallel tempering communication.
 * Handles two queues: input queue for workers and results queue from workers.
 */
class TemperingQueueManager(V) {
    private:
        JumboMessageQueue _inputQueue;
        JumboMessageQueue _resultsQueue;
        string _inputQueueName;
        string _resultsQueueName;
        
    public:
        this() {
            _inputQueueName = format("tempering_input_%d_%s",
                thisProcessID(),
                randomUUID().toString()[0..8]);
            _resultsQueueName = format("tempering_results_%d_%s",
                thisProcessID(),
                randomUUID().toString()[0..8]);
        }

        void initialize() {
            _inputQueue = new JumboMessageQueue(_inputQueueName);
            _resultsQueue = new JumboMessageQueue(_resultsQueueName);
        }

        void cleanup() {
            if (_inputQueue !is null) {
                JumboMessageQueue.cleanup(_inputQueueName);
            }
            if (_resultsQueue !is null) {
                JumboMessageQueue.cleanup(_resultsQueueName);
            }
        }

        string getInputQueueName() {
            return _inputQueueName;
        }

        string getResultsQueueName() {
            return _resultsQueueName;
        }

        JumboMessageQueue getInputQueue() {
            return _inputQueue;
        }

        JumboMessageQueue getResultsQueue() {
            return _resultsQueue;
        }
}

import io.simulation_loader : OptimizationConfig;

/// Create an optimizer based on configuration
GradientMinimizer createOptimizer(V)(
    const OptimizationConfig config, double horizon
) if (isVector!V) {
    if (config.solver_type == "gradient_descent") {
        GradientUpdateMode mode;
        if (config.gradient_descent.gradient_mode == "learning_rate")
            mode = GradientUpdateMode.LearningRate;
        else if (config.gradient_descent.gradient_mode == "step_size")
            mode = GradientUpdateMode.StepSize;
        else if (config.gradient_descent.gradient_mode == "bb1")
            mode = GradientUpdateMode.BB1;
        else if (config.gradient_descent.gradient_mode == "bb2")
            mode = GradientUpdateMode.BB2;
        else if (config.gradient_descent.gradient_mode == "bb-auto")
            mode = GradientUpdateMode.BBAuto;
        else
            throw new Exception("Invalid gradient mode: " ~ config.gradient_descent.gradient_mode);

        return createGradientDescentMinimizer!V(
            config.tolerance,
            config.max_iterations,
            config.gradient_descent.learning_rate,
            config.gradient_descent.getEffectiveGradientStep(horizon),
            mode,
            config.gradient_descent.momentum,
            config.gradient_descent.getEffectiveInitialStep(horizon),
            config.gradient_descent.getEffectiveMinStep(horizon),
            config.gradient_descent.getEffectiveMaxStep(horizon),
            totalCPUs,
            config.gradient_descent.finite_difference.order
        );
    } else if (config.solver_type == "parallel_tempering") {
        return createParallelTemperingMinimizer!V(
            config.tolerance,
            config.max_iterations,
            config.getNumReplicas(),
            config.getNumProcesses(),
            config.parallel_tempering.min_temperature,
            config.parallel_tempering.max_temperature,
            config.getEffectiveProposalStepSize(horizon)
        );
    }
    
    if (config.solver_type == "lbfgs") {
        return createLBFGSMinimizer!V(
            config.tolerance,
            config.max_iterations,
            config.lbfgs.memory_size,
            config.lbfgs.initial_step.getEffectiveValue(horizon),
            totalCPUs
        );
    }
    throw new Exception("Unknown solver type: " ~ config.solver_type);
}

/// Circular buffer for storing past optimization steps
private struct CircularBuffer(T) {
    private:
        T[] _data;
        size_t _start = 0;
        size_t _length = 0;

    public:
        this(size_t capacity) {
            _data = new T[capacity];
        }

        void push(T item) {
            if (_length < _data.length) {
                _data[_length++] = item;
            } else {
                _data[_start] = item;
                _start = (_start + 1) % _data.length;
            }
        }

        /// Access elements from newest to oldest
        const(T) opIndex(size_t index) const {
            enforce(index < _length, "Index out of bounds");
            auto pos = (_start + _length - 1 - index) % _data.length;
            return _data[pos];
        }

        size_t length() const { return _length; }
        size_t capacity() const { return _data.length; }
        void clear() { _length = 0; _start = 0; }
}

/**
 * Limited-memory BFGS optimizer as a higher-order function.
 *
 * Features:
 * - Limited memory usage via circular buffer of past updates
 * - Two-loop recursion algorithm for efficient Hessian approximation
 * - Parallel gradient computation
 * - Initial Hessian scaling
 *
 * Configuration in JSON:
 * {
 *   "solver_type": "lbfgs",
 *   "tolerance": 1e-6,
 *   "max_iterations": 1000,
 *   "lbfgs": {
 *     "memory_size": 10,      // Number of past updates to store
 *     "initial_step": {       // Initial Hessian scaling
 *       "value": 1e-4,
 *       "horizon_fraction": 0.001  // Optional
 *     },
 *     "finite_difference": {  // Configuration for gradient computation
 *       "order": 2           // Order of accuracy (2, 4, 6, or 8)
 *     }
 *   }
 * }
 */
// Implementation of L-BFGS optimization as a higher-order function
GradientMinimizer createLBFGSMinimizer(V)(
    double tolerance,
    size_t maxIterations,
    size_t memorySize,
    double initialStep,
    size_t numWorkers
) if (isVector!V) {
    // Store position and gradient differences
    struct Update {
        DynamicVector!double s;  // Position difference
        DynamicVector!double y;  // Gradient difference
        double rho;            // 1 / (y^T s)
    }
    
    auto history = CircularBuffer!Update(memorySize);
    
    // Calculate two-loop recursion direction
    DynamicVector!double twoLoopRecursion(
        const DynamicVector!double gradient,
        const CircularBuffer!Update history
    ) {
        auto q = gradient.dup;
        auto alpha = new double[history.length];
        
        // First loop
        foreach (i; 0..history.length) {
            auto update = history[i];
            alpha[i] = update.rho * update.s.dot(q);
            q = q - update.y * alpha[i];
        }
        
        // Initial Hessian approximation
        if (history.length > 0) {
            auto latest = history[0];
            double y_dot_y = latest.y.dot(latest.y);
            double scale = latest.s.dot(latest.y) / y_dot_y;
            q = q * scale;
        } else {
            q = q * initialStep;
        }
        
        // Second loop
        foreach_reverse (i; 0..history.length) {
            auto update = history[i];
            double beta = update.rho * update.y.dot(q);
            q = q + update.s * (alpha[i] - beta);
        }
        
        return q;
    }
    
    // Return the actual minimizer function
    return (const DynamicVector!double initialState, 
            ObjectiveFunction objective,
            PartialDerivativeFunction gradientFunction) {
        auto currentState = initialState.dup;
        double currentValue = objective(currentState);
        double previousValue;
        DynamicVector!double currentGradient;
        DynamicVector!double previousState;
        DynamicVector!double previousGradient;
        
        // Main optimization loop
        for (size_t iter = 0; iter < maxIterations; ++iter) {
            previousValue = currentValue;
            currentGradient = .calculateGradient!V(currentState, gradientFunction, numWorkers);
            
            if (iter > 0) {
                // Update history
                auto s = currentState - previousState;
                auto y = currentGradient - previousGradient;
                double s_dot_y = s.dot(y);
                
                // Skip update if s_dot_y is too small
                if (abs(s_dot_y) > double.epsilon * s.magnitude * y.magnitude) {
                    history.push(Update(s, y, 1.0 / s_dot_y));
                }
            }
            
            // Calculate search direction
            auto direction = twoLoopRecursion(currentGradient, history);
            
            // Line search
            double step = 1.0;
            previousState = currentState.dup;
            previousGradient = currentGradient;
            
            currentState = currentState - direction * step;
            currentValue = objective(currentState);
            
            debug {
                writefln(
                    "Gradient Magnitude: %s, Lagrangian: %s", 
                    currentGradient.magnitude, 
                    currentValue
                );
            }
            
            // Check convergence
            if (iter > 0 && abs((currentValue - previousValue) / previousValue) <= tolerance) {
                break;
            }
        }
        
        debug {
            writefln("Best energy: %s", currentValue);
        }
        
        return currentState;
    };
}

// Test gradient descent optimization
unittest {
    
    // Test different update modes
    {
        // Rastrigin function (multimodal test function)
        auto objective = delegate (const DynamicVector!double x) {
            import std.math : PI;
            double sum = 0.0;
            foreach (i; 0..x.length) {
                sum += x[i] * x[i] - 10 * cos(2 * PI * x[i]);
            }
            return 10 * x.length + sum;
        };
        
        auto x0 = DynamicVector!double([3.0, -2.0]);
        auto config = OptimizationConfig();
        config.solver_type = "gradient_descent";
        config.tolerance = 1e-6;
        config.max_iterations = 1000;
        
        // Test StepSize mode
        {
            config.gradient_descent.gradient_mode = "step_size";
            config.gradient_descent.gradient_step_size.value = 0.01;
            auto optimizer = createOptimizer!(DynamicVector!double)(config, 1.0);
            auto gradient = createFiniteDifferenceGradient(objective, 4);
            auto result = optimizer(x0, objective, gradient);
            
            // Should find local minimum
            assert(objective(result) < objective(x0), 
                "Step size mode failed to improve objective");
        }
        
        // Test BB-Auto mode
        {
            config.gradient_descent.gradient_mode = "bb-auto";
            config.gradient_descent.initial_step.value = 0.1;
            auto optimizer = createOptimizer!(DynamicVector!double)(config, 1.0);
            auto gradient = createFiniteDifferenceGradient(objective, 4);
            auto result = optimizer(x0, objective, gradient);
            
            // Should find local minimum
            assert(objective(result) < objective(x0), 
                "BB-Auto mode failed to improve objective");
        }
    }
}

// Test BB step size calculation
unittest {
    // Create test vectors
    auto state = DynamicVector!double([1.0, 2.0]);
    auto prevState = DynamicVector!double([0.0, 0.0]);
    auto grad = DynamicVector!double([2.0, 4.0]);
    auto prevGrad = DynamicVector!double([0.0, 0.0]);
    
    // Test BB1 mode
    double step = calculateBBStepSize(
        grad, state, prevGrad, prevState,
        GradientUpdateMode.BB1
    );
    assert(abs(step - 0.5) < 1e-10,  // (s^T y)/(y^T y) = 10/20 = 0.5
        "BB1 step size incorrect");
    
    // Test BB2 mode
    step = calculateBBStepSize(
        grad, state, prevGrad, prevState,
        GradientUpdateMode.BB2
    );
    assert(abs(step - 0.5) < 1e-10,  // (s^T s)/(s^T y) = 5.0/10.0 = 0.5
        "BB2 step size incorrect");
    
    // Test auto mode with oscillation (s^T y < 0)
    grad = DynamicVector!double([-2.0, -4.0]);  // Reverse gradient direction
    step = calculateBBStepSize(
        grad, state, prevGrad, prevState,
        GradientUpdateMode.BBAuto
    );
    assert(abs(step - (-0.5)) < 1e-10,  // Should choose BB2 for oscillation: (s^T s)/(s^T y) = 5.0/(-10.0) = -0.5
        "Auto mode did not choose BB2 for oscillation");
}

// Test finite difference gradient calculations
unittest {
    // Test gradient calculation on a simple quadratic function
    auto objective = delegate (const DynamicVector!double x) => x.dot(x);  // f(x) = x^T x
    
    // Create gradients with different orders
    foreach (order; [2, 4, 6, 8]) {
        auto gradient = createFiniteDifferenceGradient(objective, order);
        
        // Test at point [1, 2, 3]
        auto x = DynamicVector!double([1.0, 2.0, 3.0]);
        foreach (i; 0..3) {
            // Analytical gradient of x^T x is 2x
            double expected = 2.0 * x[i];
            double computed = gradient(x, i);
            
            // Higher order methods should be more accurate
            double tolerance = 1e-6;  // Adjust based on order
            assert(abs(computed - expected) < tolerance,
                "Gradient error too large for order " ~ order.to!string);
        }
    }
}

// Test DIFFERENCE_COEFFS properties
unittest {
    // Test coefficient sums (should be 0 for all orders)
    foreach (order; [2, 4, 6, 8]) {
        double sum = 0.0;
        foreach (coeff; DIFFERENCE_COEFFS[order]) {
            sum += coeff;
        }
        assert(abs(sum) < 1e-10, 
            "Coefficients for order " ~ order.to!string ~ " do not sum to 0");
    }
    
    // Test symmetry
    foreach (order; [2, 4, 6, 8]) {
        auto coeffs = DIFFERENCE_COEFFS[order];
        size_t n = coeffs.length;
        for (size_t i = 0; i < n/2; i++) {
            assert(abs(coeffs[i] + coeffs[n-1-i]) < 1e-10, 
                "Coefficients not antisymmetric for order " ~ order.to!string);
        }
        // Middle coefficient should be 0
        assert(abs(coeffs[n/2]) < 1e-10, 
            "Middle coefficient not 0 for order " ~ order.to!string);
    }
}

// Coefficients for finite difference calculations of different orders
public immutable double[][int] DIFFERENCE_COEFFS = [
    // 2nd order
    2: [-0.5, 0.0, 0.5],
    // 4th order
    4: [1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0],
    // 6th order
    6: [-1.0/60.0, 3.0/20.0, -3.0/4.0, 0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0],
    // 8th order
    8: [1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0]
];

// Calculate a single partial derivative using finite differences
private double calculatePartialDerivative(
    const DynamicVector!double state,
    ObjectiveFunction objective,
    size_t componentIndex,
    int finiteDifferenceOrder
) {
    // Get finite difference coefficients for current order
    auto coeffs = DIFFERENCE_COEFFS[finiteDifferenceOrder];
    int stencilLength = cast(int)coeffs.length;
    int halfPoints = (stencilLength - 1) / 2;
    
    double step_size = sqrt(double.epsilon) * max(abs(state[componentIndex]), 1);
    
    // Calculate derivative using higher-order stencil
    double derivative = 0.0;
    for (int j = 0; j < stencilLength; j++) {
        if (coeffs[j] == 0.0) continue;  // Skip center point if coefficient is 0
        
        auto eval_state = state.dup;
        eval_state[componentIndex] += step_size * (j - halfPoints);
        double eval = objective(eval_state);
        derivative += coeffs[j] * eval;
    }
    
    return derivative / step_size;
}

// Calculate gradient using parallel workers
private DynamicVector!double calculateGradient(V)(
    const DynamicVector!double state,
    PartialDerivativeFunction gradientFunction,
    size_t numWorkers
) if (isVector!V) {
    auto result = DynamicVector!double.zero(state.length);

    // Create queue for gradient components
    string queueName = format("/gradient_%d_%s", 
        thisProcessID(), 
        randomUUID().toString()[0..8]);
    auto gradientQueue = new JumboMessageQueue(queueName);
    scope(exit) JumboMessageQueue.cleanup(queueName);

    auto pids = new pid_t[numWorkers];

    // Spawn worker processes
    foreach (i; 0..numWorkers) {
        pids[i] = fork();
        if (pids[i] == 0) {
            auto queue = new JumboMessageQueue(queueName);
            
            // Process components assigned to this worker
            for (size_t idx = i; idx < state.length; idx += numWorkers) {
                double derivative = gradientFunction(state, idx);

                // Send result for this component
                auto msg = GradientMessage(i, idx, derivative);
                msg.send(queue);
            }
            exit(0);
        }
    }

    // Collect gradient components
    size_t totalComponents = state.length;
    for (size_t received = 0; received < totalComponents; received++) {
        auto msg = GradientMessage.receive(gradientQueue);
        result[msg.index] = msg.value;
    }

    // Cleanup workers
    foreach (pid; pids) {
        if (pid != 0) {
            wait(null);
        }
    }
    
    return result;
}

// Type for optimization functions
alias ObjectiveFunction = double delegate(const DynamicVector!double state);
alias PartialDerivativeFunction = double delegate(const DynamicVector!double state, size_t componentIndex);

// Creates a PartialDerivativeFunction from an ObjectiveFunction using finite differences
PartialDerivativeFunction createFiniteDifferenceGradient(
    ObjectiveFunction objective,
    int order
) {
    return (const DynamicVector!double state, size_t componentIndex) {
        // Get finite difference coefficients
        auto coeffs = DIFFERENCE_COEFFS[order];
        int stencilLength = cast(int)coeffs.length;
        int halfPoints = (stencilLength - 1) / 2;
        
        double step_size = sqrt(double.epsilon) * max(abs(state[componentIndex]), 1);
        double derivative = 0.0;
        
        for (int j = 0; j < stencilLength; j++) {
            if (coeffs[j] == 0.0) continue;
            
            auto eval_state = state.dup;
            eval_state[componentIndex] += step_size * (j - halfPoints);
            double eval = objective(eval_state);
            derivative += coeffs[j] * eval;
        }
        
        return derivative / step_size;
    };
}

// Function types that perform minimization
alias Minimizer = DynamicVector!double delegate(
    const DynamicVector!double initialState,
    ObjectiveFunction objective
);

alias GradientMinimizer = DynamicVector!double delegate(
    const DynamicVector!double initialState,
    ObjectiveFunction objective,          // For energy evaluation (convergence, line search)
    PartialDerivativeFunction gradientFunction  // For gradient computation
);

/**
 * Parallel Tempering solver for non-convex optimization problems.
 *
 * This solver maintains multiple replicas of the system at different temperatures,
 * allowing for both local optimization (cold replicas) and global exploration
 * (hot replicas). Features:
 *
 * - Monte Carlo sampling at each temperature level
 * - Temperature reassignment based on energy ordering
 * - Message queue-based communication between replicas
 * - Energy-based replica swapping
 * - Adaptive step sizes based on temperature
 * - Automatic cleanup of child processes and queues
 *
 * Configuration in JSON:
 * {
 *   "solver_type": "parallel_tempering",
 *   "tolerance": 1e-6,
 *   "max_iterations": 1000,
 *   "parallel_tempering": {
 *     "num_replicas": 8,  // Optional: defaults to CPU count
 *     "num_processes": 4,  // Optional: defaults to CPU count
 *     "min_temperature": 0.1,         // Controls local optimization
 *     "max_temperature": 2.0,         // Controls global exploration
 *     "proposal_step_size": {
 *       "value": 1e-4,                // Fixed step size for proposals
 *       "horizon_fraction": 0.0001    // Optional: step = horizon * fraction
 *     }
 *   }
 * }
 */
// Implementation of parallel tempering optimization as a higher-order function
GradientMinimizer createParallelTemperingMinimizer(V)(
    double tolerance,
    size_t maxIterations,
    size_t numReplicas,
    size_t numProcesses,
    double minTemperature,
    double maxTemperature,
    double proposalStepSize
) if (isVector!V) {
    // Helper struct for tracking best solution
    static struct BestSolution {
        DynamicVector!double state;
        double energy = double.infinity;

        this(DynamicVector!double s, double e = double.infinity) {
            state = s.dup;
            energy = e;
        }
    }

    // Helper functions for managing temperatures
    double[] initializeTemperatures() {
        import std.math : pow;
        auto temperatures = new double[numReplicas];
        double ratio = pow(maxTemperature / minTemperature, 1.0 / (numReplicas - 1));
        foreach (i; 0..numReplicas) {
            temperatures[i] = minTemperature * pow(ratio, i);
        }
        return temperatures;
    }
        
    // Sort states by energy and return reordered temperatures
    double[] reorderTemperatures(double[] temperatures, double[] energies) {
        import std.algorithm : sort;
        
        // Collect energies and indices
        struct StateEnergy {
            size_t stateIndex;
            double energy;
        }
        
        auto allStates = new StateEnergy[temperatures.length];
        foreach (i; 0..temperatures.length) {
            allStates[i] = StateEnergy(i, energies[i]);
        }
        
        // Sort by energy
        sort!((a, b) => a.energy < b.energy)(allStates);
        
        // Create new temperature assignments
        auto newTemps = new double[temperatures.length];
        foreach (i; 0..temperatures.length) {
            newTemps[allStates[i].stateIndex] = temperatures[i];
        }
        
        return newTemps;
    }
    
    // Helper function for worker process management
    void handleWorkerProcess(
        ObjectiveFunction objective,
        string inputQueueName,
        string resultsQueueName,
        double stepSize
    ) {
        try {
            // Connect to queues
            auto inputQueue = new JumboMessageQueue(inputQueueName);
            auto resultsQueue = new JumboMessageQueue(resultsQueueName);
            
            // Process jobs until terminated
            while (true) {
                // Get current state and temperature
                auto input = WorkerInput!V.receive(inputQueue);
                
                // Generate proposal with random walk
                auto proposedState = input.state.dup;
                foreach (i; 0..proposedState.length) {
                    proposedState[i] += normal(0, stepSize);
                }
                
                // Evaluate proposal
                double proposedEnergy = objective(proposedState);
                
                // Handle Metropolis acceptance
                bool accepted = proposedEnergy <= input.energy || 
                    uniform01() < exp((input.energy - proposedEnergy) / input.temperature);
                
                // Return result
                WorkerResult!V result;
                if (accepted) {
                    result = WorkerResult!V(proposedState, proposedEnergy);
                } else {
                    result = WorkerResult!V(input.state, input.energy);
                }
                result.send(resultsQueue);
            }
        } catch (Exception e) {
            stderr.writefln("Worker error: %s", e.msg);
        }
    }

    // Return the actual minimizer function
    return (const DynamicVector!double initialState,
            ObjectiveFunction objective,
            PartialDerivativeFunction gradientFunction) { // Added gradientFunction parameter
        // Initialize queue manager
        auto queueManager = new TemperingQueueManager!V();
        scope(exit) queueManager.cleanup();

        // Create worker process manager
        auto processManager = new ProcessManager(numProcesses);
        scope(exit) processManager.cleanupProcesses();

        // Setup queues
        queueManager.initialize();
        
        // Initialize temperatures and states
        auto replicaTemperatures = initializeTemperatures();
        auto replicaConfigurations = new DynamicVector!double[numReplicas];
        auto replicaEnergies = new double[numReplicas];
        
        // Initialize all replicas
        foreach (i; 0..numReplicas) {
            replicaConfigurations[i] = initialState.dup;
            replicaEnergies[i] = objective(replicaConfigurations[i]);
        }

        // Spawn workers
        processManager.spawnWorkers((size_t workerId) {
            handleWorkerProcess(
                objective,
                queueManager.getInputQueueName(),
                queueManager.getResultsQueueName(),
                proposalStepSize
            );
        });
        
        // Track best solution
        auto best = BestSolution(initialState.dup, double.infinity);
        
        // Main optimization loop
        for (size_t iter = 0; iter < maxIterations; ++iter) {
            // Send current state to workers
            foreach (i; 0..numReplicas) {
                auto input = WorkerInput!V(
                    replicaConfigurations[i],
                    replicaEnergies[i],
                    replicaTemperatures[i]
                );
                input.send(queueManager.getInputQueue());
            }

            // Collect results
            foreach (i; 0..numReplicas) {
                auto result = WorkerResult!V.receive(queueManager.getResultsQueue());
                
                // Update replica state
                replicaConfigurations[i] = result.state;
                replicaEnergies[i] = result.energy;
                
                // Update best solution if needed
                if (result.energy < best.energy) {
                    best = BestSolution(result.state, result.energy);
                }
            }
            
            // Reorder temperatures
            replicaTemperatures = reorderTemperatures(replicaTemperatures, replicaEnergies);
        }

        debug {
            writefln("Best energy: %s", best.energy);
        }
        
        return best.state;
    };
}

/**
 * Update modes for gradient descent optimization:
 * - LearningRate: Scale gradient by learning rate (traditional gradient descent)
 * - StepSize: Move fixed distance in gradient direction (normalized gradient)
 * - BB1: Barzilai-Borwein method 1 (s^T y / y^T y)
 * - BB2: Barzilai-Borwein method 2 (s^T s / s^T y)
 * - BBAuto: Automatic selection between BB1 and BB2
 */
enum GradientUpdateMode {
    LearningRate,  // Scale gradient by learning rate
    StepSize,      // Move fixed step size in gradient direction
    BB1,           // Barzilai-Borwein method 1
    BB2,           // Barzilai-Borwein method 2
    BBAuto         // Automatic BB method selection
}

// Gradient message for parallel computation - one component at a time
private struct GradientMessage {
    size_t workerId;          // Worker identifier
    size_t index;            // Component index in optimization state
    double value;            // Gradient value for this component

    ubyte[] toBytes() {
        ubyte[] buffer;
        buffer ~= cast(ubyte[])(&workerId)[0..1];
        buffer ~= cast(ubyte[])(&index)[0..1];
        buffer ~= cast(ubyte[])(&value)[0..1];
        return buffer;
    }

    static GradientMessage fromBytes(ubyte[] data) {
        GradientMessage msg;
        size_t offset = 0;
        
        msg.workerId = *cast(size_t*)(data.ptr + offset);
        offset += size_t.sizeof;
        
        msg.index = *cast(size_t*)(data.ptr + offset);
        offset += size_t.sizeof;
        
        msg.value = *cast(double*)(data.ptr + offset);
        
        return msg;
    }

    void send(JumboMessageQueue queue) {
        queue.send(this.toBytes());
    }

    static GradientMessage receive(JumboMessageQueue queue) {
        return fromBytes(queue.receive());
    }
}

/**
 * Gradient descent solver with momentum and parallel gradient computation.
 *
 * Features:
 * - Momentum-based updates for faster convergence
 * - Parallel gradient computation across multiple processes
 * - Support for multiple gradient update modes:
 *   - Learning rate mode (traditional gradient descent)
 *   - Fixed step size mode (normalized gradient)
 *   - Barzilai-Borwein methods (adaptive step size)
 * - Automatic gradient normalization
 * 
 * Configuration in JSON:
 * {
 *   "solver_type": "gradient_descent",
 *   "tolerance": 1e-6,
 *   "max_iterations": 1000,
 *   "gradient_descent": {
 *     "momentum": 0.9,
 *     "gradient_mode": "step_size",  // "step_size", "learning_rate", "bb1", "bb2", "bb-auto"
 *     "learning_rate": 0.01,  // Used if mode is "learning_rate"
 *     "finite_difference_step": 1e-6,  // Step size for numerical derivatives
 *     
 *     // Step size configurations (all support horizon_fraction)
 *     "initial_step": {  // Initial step size for BB methods
 *       "value": 1e-4,
 *       "horizon_fraction": 0.001
 *     },
 *     "min_step": {  // Minimum allowed step size
 *       "value": 1e-10,
 *       "horizon_fraction": 0.00001
 *     },
 *     "max_step": {  // Maximum allowed step size
 *       "value": 1e-2,
 *       "horizon_fraction": 0.01
 *     },
 *     "gradient_step_size": {  // Step size for fixed step mode
 *       "value": 1e-4,
 *       "horizon_fraction": 0.0001
 *     }
 *   }
 * }
 *
 * Barzilai-Borwein Methods:
 * - BB1: Uses step size α_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
 * - BB2: Uses step size α_k = (s_{k-1}^T s_{k-1}) / (s_{k-1}^T y_{k-1})
 * - BB-Auto: Automatically switches between BB1 and BB2 based on convergence behavior
 *   - Uses BB2 when oscillating (s^T y < 0)
 *   - Uses BB1 otherwise (s^T y ≥ 0)
 * where:
 * - s_{k-1} = x_k - x_{k-1} (change in position)
 * - y_{k-1} = g_k - g_{k-1} (change in gradient)
 */
// Implementation of gradient descent optimization as a higher-order function
GradientMinimizer createGradientDescentMinimizer(V)(
    double tolerance,
    size_t maxIterations,
    double learningRate,
    double stepSize,
    GradientUpdateMode mode,
    double momentum,
    double initialStep,
    double minStep,
    double maxStep,
    size_t numWorkers,
    int finiteDifferenceOrder
) if (isVector!V) {
        
    // Helper function to calculate BB step size
    double calculateBBStepSize(
        const DynamicVector!double gradient,
        const DynamicVector!double state,
        const DynamicVector!double previousGradient,
        const DynamicVector!double previousState,
        GradientUpdateMode updateMode
    ) {
        auto s = state - previousState;          // Change in position
        auto y = gradient - previousGradient;    // Change in gradient
        
        double s_dot_y = s.dot(y);  // s^T y
        double y_dot_y = y.dot(y);  // y^T y
        double s_dot_s = s.dot(s);  // s^T s
        
        double bb1 = s_dot_y / y_dot_y;  // BB1 step size
        double bb2 = s_dot_s / s_dot_y;  // BB2 step size
        
        //enforce(!isNaN(bb1) && !isNaN(bb2), "Invalid BB step sizes");
        if (isNaN(bb1) && isNaN(bb2)) {
            return 0;
        }
        else if (isNaN(bb1)) {
            return bb2;
        }
        else if (isNaN(bb2)) {
            return bb1;
        }
        
        // Auto mode selection based on which step gives better descent direction
        if (updateMode == GradientUpdateMode.BBAuto) {
            // Choose BB2 if oscillating (indicated by negative s_dot_y)
            return (s_dot_y < 0) ? bb2 : bb1;
        }
        else if (updateMode == GradientUpdateMode.BB1) {
            return bb1;
        }
        else if (updateMode == GradientUpdateMode.BB2) {
            return bb2;
        }
        else {
            assert(0, "Invalid update mode");
        }
    }
        
    // Return the actual minimizer function
    return (const DynamicVector!double initialState,
            ObjectiveFunction objective,
            PartialDerivativeFunction gradientFunction) {
        // Initialize state with velocity
        DynamicVector!double state = initialState.dup;
        DynamicVector!double velocity = DynamicVector!double.zero(initialState.length);

        DynamicVector!double previousGradient;
        DynamicVector!double previousState;
        
        double value;
        
        // Main optimization loop
        for (size_t iter = 0; iter < maxIterations; ++iter) {
            
            // Calculate gradients
            auto gradient = .calculateGradient!V(state, gradientFunction, numWorkers);
            
            // Compute total gradient magnitude
            double totalMagnitude = sqrt(gradient.magnitudeSquared());
            
            double step;
            DynamicVector!double updateDirection;

            if (mode == GradientUpdateMode.LearningRate) {
                step = learningRate;
                updateDirection = gradient;
            }
            else if (mode == GradientUpdateMode.StepSize) {
                step = stepSize;
                updateDirection = gradient / totalMagnitude;  // Normalize
            }
            else {  // BB modes
                if (iter == 0) {
                    // Use initial step size if no previous state
                    step = initialStep;
                } else {
                    // Calculate step size using BB method
                    step = calculateBBStepSize(gradient, state, previousGradient, previousState, mode);
                }
                updateDirection = gradient;
            }
            
            previousGradient = gradient;
            previousState = state;

            // Update velocity using momentum
            velocity = velocity * momentum + updateDirection * (1 - momentum);
            state = state - velocity * min(max(step, minStep), maxStep);

            // Evaluate current state
            double previousValue = value;
            value = objective(state);

            debug {
                writefln(
                    "Gradient Magnitude: %s, Step size: %s, Lagrangian: %s", 
                    totalMagnitude, 
                    step,
                    value
                );
            }

            // Check convergence
            if (iter > 0 && (abs(value - previousValue) / previousValue) <= tolerance) {
                break;
            }
        }
            
        debug {
            writefln("Best energy: %s", value);
        }
        
        return state;
    };
}

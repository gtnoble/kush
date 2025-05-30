module core.optimization;

import core.material_point;
import math.vector;
import std.math : abs, sqrt, exp;
import std.math.traits : isNaN;
import std.algorithm : min, max, map, sum;
import core.thread : Thread;
import core.time : dur;
import std.array;
import std.random : uniform01;
import std.mathspecial : normalDistributionInverse;
import std.stdio;
import core.sys.posix.unistd : fork, pid_t, close;
import core.sys.posix.signal : 
    kill, sigaction, sigaction_t,
    SIGTERM, SIGABRT, SIGINT, SA_RESTART;
import core.sys.posix.sys.wait : wait, WIFEXITED, WIFSIGNALED, WEXITSTATUS, WTERMSIG;
import core.stdc.stdlib : exit;
import std.conv : to;
import std.format : format;
import core.stdc.errno;
import core.stdc.string : strerror;
import std.parallelism : totalCPUs;
import std.process : thisProcessID;
import std.string : fromStringz;
import std.uuid : randomUUID;
import std.file : remove;
import std.exception : enforce;
import jumbomessage;

enum ReadyStatus {
    ReadyToRead = 0,
    ReadyToWrite = 1,
    ReadyToReadWrite = 2,
    NotReady = 3
}


// Helper for normal distribution sampling
private double normal(double mean, double stddev) {
    return mean + stddev * normalDistributionInverse(uniform01());
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

/// Unified state type that encapsulates both positions and multipliers
struct OptimizationState(V) {
    V[] positions;  // Current positions
    DynamicVector!double multipliers;  // Vector of multipliers for constraints

    // Total number of scalar components (positions * V.length + multipliers.length)
    size_t numComponents() const {
        return positions.length * V.length + multipliers.length;
    }

    // Get component by linear index
    double opIndex(size_t index) const {
        if (index < positions.length * V.length) {
            size_t pointIndex = index / V.length;
            size_t dimIndex = index % V.length;
            return positions[pointIndex][dimIndex];
        }
        return multipliers[index - positions.length * V.length];
    }

    // Set component by linear index
    double opIndexOpAssign(string op)(double value, size_t index)
        if (op == "+" || op == "-" || op == "*" || op == "/")
    {
        if (index < positions.length * V.length) {
            size_t pointIndex = index / V.length;
            size_t dimIndex = index % V.length;
            static if (op == "+") 
                positions[pointIndex][dimIndex] += value;
            else static if (op == "-") 
                positions[pointIndex][dimIndex] -= value;
            else static if (op == "*") 
                positions[pointIndex][dimIndex] *= value;
            else static if (op == "/") 
                positions[pointIndex][dimIndex] /= value;
        } else {
            size_t multiplierIndex = index - positions.length * V.length;
            static if (op == "+") 
                multipliers[multiplierIndex] += value;
            else static if (op == "-") 
                multipliers[multiplierIndex] -= value;
            else static if (op == "*") 
                multipliers[multiplierIndex] *= value;
            else static if (op == "/") 
                multipliers[multiplierIndex] /= value;
        }
        return opIndex(index);
    }

    // Basic assignment
    double opIndexAssign(double value, size_t index) {
        if (index < positions.length * V.length) {
            size_t pointIndex = index / V.length;
            size_t dimIndex = index % V.length;
            positions[pointIndex][dimIndex] = value;
        } else {
            multipliers[index - positions.length * V.length] = value;
        }
        return value;
    }

    this(V[] pos, size_t numConstraints) {
        positions = pos;
        multipliers = DynamicVector!double(numConstraints);
    }
    
    this(V[] pos, DynamicVector!double mult) {
        positions = pos;
        multipliers = mult;
    }

    OptimizationState!V dup() const {
        auto result = OptimizationState!V(positions.dup, multipliers.dup);
        return result;
    }

    OptimizationState!V opBinary(string op)(const OptimizationState!V other) const
        if (op == "+" || op == "-")
    {
        assert(numComponents == other.numComponents, "Dimension mismatch");
        auto result = OptimizationState!V(new V[positions.length], multipliers.length);
        static if (op == "+") {
            foreach (i; 0..positions.length) {
                result.positions[i] = positions[i] + other.positions[i];
            }
            result.multipliers = multipliers + other.multipliers;
        } else static if (op == "-") {
            foreach (i; 0..positions.length) {
                result.positions[i] = positions[i] - other.positions[i];
            }
            result.multipliers = multipliers - other.multipliers;
        }
        return result;
    }

    OptimizationState!V opBinary(string op)(double scalar) const
        if (op == "*" || op == "/")
    {
        auto result = OptimizationState!V(new V[positions.length], multipliers.length);
        static if (op == "*") {
            foreach (i; 0..positions.length) {
                result.positions[i] = positions[i] * scalar;
            }
            result.multipliers = multipliers * scalar;
        } else static if (op == "/") {
            foreach (i; 0..positions.length) {
                result.positions[i] = positions[i] / scalar;
            }
            result.multipliers = multipliers / scalar;
        }
        return result;
    }

    double magnitudeSquared() const {
        double total = 0.0;
        foreach (pos; positions) {
            total += pos.magnitudeSquared();
        }
        total += multipliers.magnitudeSquared();
        return total;
    }
    
    double magnitude() const {
        return sqrt(magnitudeSquared());
    }

    // Calculate dot product with another state
    double dot(const OptimizationState!V other) const {
        double result = 0.0;
        foreach (i; 0..numComponents) {
            result += this[i] * other[i];
        }
        return result;
    }

    static OptimizationState!V zero(size_t numPositions, size_t numConstraints = 0) {
        auto result = OptimizationState!V(new V[numPositions], DynamicVector!double.zero(numConstraints));
        foreach (ref pos; result.positions) {
            pos = V.zero();
        }
        // multipliers initialized to zero by default
        return result;
    }

    ubyte[] toBytes() const {
        ubyte[] buffer;
        
        // Add canary value
        buffer ~= cast(ubyte)42;
        
        // Serialize positions array size
        size_t size = positions.length;
        buffer ~= cast(ubyte[])(&size)[0..1];
        
        // Serialize each position vector
        foreach (pos; positions) {
            buffer ~= pos.toBytes();
        }
        
        // Serialize multipliers vector
        buffer ~= multipliers.toBytes();
        
        return buffer;
    }

    void send(JumboMessageQueue queue) const {
        queue.send(this.toBytes());
    }
        
    static size_t fromBytes(ubyte[] data, ref OptimizationState!V target) {
        size_t offset = 0;
        
        // Check canary value
        enforce(data[0] == 42, "Invalid canary value in serialized OptimizationState");
        offset += 1;
        
        // Read positions array size
        size_t size = *cast(size_t*)(data.ptr + offset);
        offset += size_t.sizeof;
        
        // Create new positions array and deserialize
        auto positions = new V[size];
        foreach (i; 0..size) {
            offset += V.fromBytes(data[offset..$], positions[i]);
        }
        
        // Create and deserialize new multipliers
        auto multipliers = DynamicVector!double(0);
        offset += DynamicVector!double.fromBytes(data[offset..$], multipliers);
        
        // Create new state using constructor
        target = OptimizationState!V(positions, multipliers);
        
        return offset;
    }

    static OptimizationState!V fromBytes(ubyte[] data) {
        OptimizationState!V result;
        fromBytes(data, result);
        return result;
    }

    static OptimizationState!V receive(JumboMessageQueue queue) {
        return fromBytes(queue.receive());
    }
}

// Replica configuration now uses the unified state type
private alias ReplicaConfiguration(V) = OptimizationState!V;

private struct WorkerInput(V) {
    OptimizationState!V state;
    double energy;
    double temperature;

    ubyte[] toBytes() {
        ubyte[] buffer;

        // Serialize current state
        auto stateBytes = state.toBytes();
        buffer ~= stateBytes;

        // Serialize energy and temperature
        buffer ~= cast(ubyte[])(&energy)[0..1];
        buffer ~= cast(ubyte[])(&temperature)[0..1];

        return buffer;
    }

    static WorkerInput!V fromBytes(ubyte[] data) {
        size_t offset = 0;
        OptimizationState!V state;
        
        // Deserialize state using reference method
        offset += OptimizationState!V.fromBytes(data, state);
        
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
    OptimizationState!V state;
    double energy;

    ubyte[] toBytes() {
        ubyte[] buffer;
        
        // Serialize state
        auto stateBytes = state.toBytes();
        buffer ~= stateBytes;

        // Serialize energy
        buffer ~= cast(ubyte[])(&energy)[0..1];

        return buffer;
    }

    static WorkerResult!V fromBytes(ubyte[] data) {
        size_t offset = 0;
        OptimizationState!V state;
        
        // Deserialize state using reference method
        offset += OptimizationState!V.fromBytes(data, state);
        
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
Minimizer!V createOptimizer(V)(
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
            totalCPUs,
            config.lbfgs.finite_difference.order
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
Minimizer!V createLBFGSMinimizer(V)(
    double tolerance,
    size_t maxIterations,
    size_t memorySize,
    double initialStep,
    size_t numWorkers,
    int finiteDifferenceOrder
) if (isVector!V) {
    // Store position and gradient differences
    struct Update {
        OptimizationState!V s;  // Position difference
        OptimizationState!V y;  // Gradient difference
        double rho;            // 1 / (y^T s)
    }
    
    auto history = CircularBuffer!Update(memorySize);
    
    // Calculate two-loop recursion direction
    OptimizationState!V twoLoopRecursion(
        const OptimizationState!V gradient,
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
    return (const OptimizationState!V initialState, ObjectiveFunction!V objective) {
        auto currentState = initialState.dup;
        double currentValue = objective(currentState);
        double previousValue;
        
        OptimizationState!V currentGradient;
        OptimizationState!V previousState;
        OptimizationState!V previousGradient;
        
        // Main optimization loop
        for (size_t iter = 0; iter < maxIterations; ++iter) {
            previousValue = currentValue;
            currentGradient = .calculateGradient!V(currentState, objective, numWorkers, finiteDifferenceOrder);
            
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
                    "Gradient Magnitude: %s, Position Partial Gradient: %s, Multiplier Value: %s, Multiplier Partial Gradient: %s, Lagrangian: %s", 
                    currentGradient.magnitude, 
                    currentGradient.positions.map!((value) => value.magnitudeSquared).sum.sqrt,
                    currentState.multipliers.magnitude,
                    currentGradient.multipliers.magnitude,
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

// Coefficients for finite difference calculations of different orders
private immutable double[][int] DIFFERENCE_COEFFS = [
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
private double calculatePartialDerivative(V)(
    const OptimizationState!V state,
    ObjectiveFunction!V objective,
    size_t componentIndex,
    int finiteDifferenceOrder
) if (isVector!V) {
    // Get finite difference coefficients for current order
    auto coeffs = DIFFERENCE_COEFFS[finiteDifferenceOrder];
    int stencilLength = cast(int)coeffs.length;
    int halfPoints = (stencilLength - 1) / 2;
    
    double step_size = sqrt(double.epsilon) * max(abs(state[componentIndex]), 1);
    
    // Calculate derivative using higher-order stencil
    double derivative = 0.0;
    for (int j = 0; j < stencilLength; j++) {
        if (coeffs[j] == 0.0) continue;  // Skip center point if coefficient is 0
        
        auto eval_state = state.dup();
        eval_state[componentIndex] += step_size * (j - halfPoints);
        double eval = objective(eval_state);
        derivative += coeffs[j] * eval;
    }
    
    return derivative / step_size;
}

// Calculate gradient using parallel workers
private OptimizationState!V calculateGradient(V)(
    const OptimizationState!V state,
    ObjectiveFunction!V objective,
    size_t numWorkers,
    int finiteDifferenceOrder = 2
) if (isVector!V) {
    const size_t numPoints = state.positions.length;
    const size_t numConstraints = state.multipliers.length;
    auto result = OptimizationState!V.zero(numPoints, numConstraints);

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
            for (size_t idx = i; idx < state.numComponents; idx += numWorkers) {
                double derivative = calculatePartialDerivative!V(state, objective, idx, finiteDifferenceOrder);

                // Send result for this component
                auto msg = GradientMessage(i, idx, derivative);
                msg.send(queue);
            }
            exit(0);
        }
    }

    // Collect gradient components
    size_t totalComponents = state.numComponents;
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

// Type for optimization objective functions
alias ObjectiveFunction(V) = double delegate(const OptimizationState!V state);

// Function type that performs actual minimization
alias Minimizer(V) = OptimizationState!V delegate(
    const OptimizationState!V initialState,
    ObjectiveFunction!V objective
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
Minimizer!V createParallelTemperingMinimizer(V)(
    double tolerance,
    size_t maxIterations,
    size_t numReplicas,
    size_t numProcesses,
    double minTemperature,
    double maxTemperature,
    double proposalStepSize
) if (isVector!V) {
    // Helper struct for tracking best solution
    static struct BestSolution(V) {
        OptimizationState!V state;
        double energy = double.infinity;

        this(OptimizationState!V s, double e = double.infinity) {
            state = s.dup();
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
        ObjectiveFunction!V objective,
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
                auto proposedState = input.state.dup();
                foreach (i; 0..proposedState.numComponents) {
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
    return (const OptimizationState!V initialState, ObjectiveFunction!V objective) {
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
        auto replicaConfigurations = new ReplicaConfiguration!V[numReplicas];
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
        auto best = BestSolution!V(initialState.dup, double.infinity);
        
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
                    best = BestSolution!V(result.state, result.energy);
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
Minimizer!V createGradientDescentMinimizer(V)(
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
        const OptimizationState!V gradient,
        const OptimizationState!V state,
        const OptimizationState!V previousGradient,
        const OptimizationState!V previousState,
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
    return (const OptimizationState!V initialState, ObjectiveFunction!V objective) {
        // Initialize state with velocity
        OptimizationState!V state = initialState.dup;
        OptimizationState!V velocity = OptimizationState!V.zero(initialState.positions.length, initialState.multipliers.length);

        OptimizationState!V previousGradient;
        OptimizationState!V previousState;
        
        double value;
        
        // Main optimization loop
        for (size_t iter = 0; iter < maxIterations; ++iter) {
            
            // Calculate gradients
            auto gradient = .calculateGradient!V(state, objective, numWorkers, finiteDifferenceOrder);
            
            // Compute total gradient magnitude
            double totalMagnitude = sqrt(gradient.magnitudeSquared());
            
            double step;
            OptimizationState!V updateDirection;

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
                    "Gradient Magnitude: %s, Step size: %s, Position Partial Gradient: %s, Multiplier Value: %s, Multiplier Partial Gradient: %s, Lagrangian: %s", 
                    totalMagnitude, 
                    step,
                    gradient.positions.map!((value) => value.magnitudeSquared).sum.sqrt,
                    state.multipliers.magnitude,
                    gradient.multipliers.magnitude,
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

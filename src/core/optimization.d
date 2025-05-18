module core.optimization;

import core.material_point;
import math.vector;
import std.math : abs, sqrt, exp;
import std.algorithm : min, max;
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

/// Unified state type that encapsulates both positions and multiplier
struct OptimizationState(V) {
    V[] positions;  // Current positions
    double multiplier;  // Scalar multiplier for velocity constraints

    // Total number of scalar components (positions * V.dimension + 1 multiplier)
    size_t numComponents() const {
        return positions.length * V.dimension + 1;
    }

    // Get component by linear index
    double opIndex(size_t index) const {
        if (index < positions.length * V.dimension) {
            size_t pointIndex = index / V.dimension;
            size_t dimIndex = index % V.dimension;
            return positions[pointIndex][dimIndex];
        }
        return multiplier;
    }

    // Set component by linear index
    double opIndexOpAssign(string op)(double value, size_t index)
        if (op == "+" || op == "-" || op == "*" || op == "/")
    {
        if (index < positions.length * V.dimension) {
            size_t pointIndex = index / V.dimension;
            size_t dimIndex = index % V.dimension;
            static if (op == "+") 
                positions[pointIndex][dimIndex] += value;
            else static if (op == "-") 
                positions[pointIndex][dimIndex] -= value;
            else static if (op == "*") 
                positions[pointIndex][dimIndex] *= value;
            else static if (op == "/") 
                positions[pointIndex][dimIndex] /= value;
        } else {
            static if (op == "+") 
                multiplier += value;
            else static if (op == "-") 
                multiplier -= value;
            else static if (op == "*") 
                multiplier *= value;
            else static if (op == "/") 
                multiplier /= value;
        }
        return opIndex(index);
    }

    // Basic assignment
    double opIndexAssign(double value, size_t index) {
        if (index < positions.length * V.dimension) {
            size_t pointIndex = index / V.dimension;
            size_t dimIndex = index % V.dimension;
            positions[pointIndex][dimIndex] = value;
        } else {
            multiplier = value;
        }
        return value;
    }

    this(V[] pos, double mult = 0.0) {
        positions = pos;
        multiplier = mult;
    }

    OptimizationState!V dup() const {
        return OptimizationState!V(positions.dup, multiplier);
    }

    OptimizationState!V opBinary(string op)(const OptimizationState!V other) const
        if (op == "+" || op == "-")
    {
        assert(numComponents == other.numComponents, "Dimension mismatch");
        auto result = OptimizationState!V(new V[positions.length]);
        static if (op == "+") {
            foreach (i; 0..positions.length) {
                result.positions[i] = positions[i] + other.positions[i];
            }
            result.multiplier = multiplier + other.multiplier;
        } else static if (op == "-") {
            foreach (i; 0..positions.length) {
                result.positions[i] = positions[i] - other.positions[i];
            }
            result.multiplier = multiplier - other.multiplier;
        }
        return result;
    }

    OptimizationState!V opBinary(string op)(double scalar) const
        if (op == "*" || op == "/")
    {
        auto result = OptimizationState!V(new V[positions.length]);
        static if (op == "*") {
            foreach (i; 0..positions.length) {
                result.positions[i] = positions[i] * scalar;
            }
            result.multiplier = multiplier * scalar;
        } else static if (op == "/") {
            foreach (i; 0..positions.length) {
                result.positions[i] = positions[i] / scalar;
            }
            result.multiplier = multiplier / scalar;
        }
        return result;
    }

    double magnitudeSquared() const {
        double total = multiplier * multiplier;  // Include multiplier in magnitude
        foreach (pos; positions) {
            total += pos.magnitudeSquared();
        }
        return total;
    }

    static OptimizationState!V zero(size_t numPositions) {
        auto result = OptimizationState!V(new V[numPositions]);
        foreach (ref pos; result.positions) {
            pos = V.zero();
        }
        result.multiplier = 0.0;
        return result;
    }

    ubyte[] toBytes() const {
        ubyte[] buffer;
        buffer ~= cast(ubyte[])(&multiplier)[0..1];
        
        // Serialize positions array size and data
        size_t size = positions.length;
        buffer ~= cast(ubyte[])(&size)[0..1];
        buffer ~= cast(ubyte[])positions[];
        
        return buffer;
    }

    void send(JumboMessageQueue queue) const {
        queue.send(this.toBytes());
    }
        
    static OptimizationState!V fromBytes(ubyte[] data) {
        OptimizationState!V state;
        size_t offset = 0;
        
        // Deserialize multiplier
        state.multiplier = *cast(double*)(data.ptr + offset);
        offset += double.sizeof;
        
        // Deserialize positions array size and data
        size_t size = *cast(size_t*)(data.ptr + offset);
        offset += size_t.sizeof;
        
        state.positions = new V[size];
        state.positions[] = cast(V[])(data[offset..offset + V.sizeof * size])[];
        
        return state;
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

        // Deserialize state
        auto state = OptimizationState!V.fromBytes(data);
        offset += state.toBytes().length;

        // Deserialize energy and temperature
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

        // Deserialize state
        auto state = OptimizationState!V.fromBytes(data);
        offset += state.toBytes().length;

        // Deserialize energy
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

// Result type containing positions and scalar Lagrange multiplier
struct OptimizationResult(T, V) if (isMaterialPoint!(T, V)) {
    V[] positions;
    double multiplier;    // Single scalar multiplier for all velocity constraints
    
    this(V[] pos, double mult = 0.0) {
        positions = pos;
        multiplier = mult;
    }
}

import io.simulation_loader : OptimizationConfig;

/// Create an optimizer based on configuration
OptimizationSolver!(T, V) createOptimizer(T, V)(
    const OptimizationConfig config, double horizon
) if (isMaterialPoint!(T, V)) {
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

        return new GradientDescentSolver!(T, V)(
            config.tolerance,
            config.max_iterations,
            config.gradient_descent.learning_rate,
            config.gradient_descent.getEffectiveGradientStep(horizon),
            mode,
            config.gradient_descent.momentum,
            config.gradient_descent.finite_difference_step,
            config.gradient_descent.getEffectiveInitialStep(horizon),
            config.gradient_descent.getEffectiveMinStep(horizon),
            config.gradient_descent.getEffectiveMaxStep(horizon)
        );
    } else if (config.solver_type == "parallel_tempering") {
        return new ParallelTemperingSolver!(T, V)(
            config.tolerance,
            config.max_iterations,
            config.getNumReplicas(),
            config.getNumProcesses(),
            config.parallel_tempering.min_temperature,
            config.parallel_tempering.max_temperature,
            config.getEffectiveProposalStepSize(horizon)
        );
    }
    
    throw new Exception("Unknown solver type: " ~ config.solver_type);
}

// Interface for optimization objective functions
interface ObjectiveFunction(T, V) if (isMaterialPoint!(T, V)) {
    // Evaluate objective function with unified state
    double evaluate(OptimizationState!V state);
}

// Base class for optimization-based solvers
abstract class OptimizationSolver(T, V) if (isMaterialPoint!(T, V)) {
    protected:
        double _tolerance;
        size_t _maxIterations;
        
    public:
        this(double tolerance = 1e-6, size_t maxIterations = 1000) {
            _tolerance = tolerance;
            _maxIterations = maxIterations;
        }
        
        // Core optimization method to be implemented by concrete solvers
        abstract OptimizationResult!(T, V) minimize(
            OptimizationState!V initialState,
            ObjectiveFunction!(T, V) objective
        );
}

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
class ParallelTemperingSolver(T, V) : OptimizationSolver!(T, V) {
    private:
        static struct BestSolution(V) {
            OptimizationState!V state;
            double energy = double.infinity;

            this(OptimizationState!V s, double e = double.infinity) {
                state = s.dup();
                energy = e;
            }
        }

        size_t _numReplicas;
        size_t _numProcesses;
        double _minTemperature;
        double _maxTemperature;
        double _proposalStepSize = 1e-6;
        
        // Resource management
        ProcessManager _processManager;
        
        // Initialize replica temperatures using geometric progression
        double[] initializeTemperatures() {
            import std.math : pow;
            auto temperatures = new double[_numReplicas];
            double ratio = pow(_maxTemperature / _minTemperature, 1.0 / (_numReplicas - 1));
            foreach (i; 0.._numReplicas) {
                temperatures[i] = _minTemperature * pow(ratio, i);
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
            
            auto allStates = new StateEnergy[_numReplicas];
            foreach (i; 0.._numReplicas) {
                allStates[i] = StateEnergy(i, energies[i]);
            }
            
            // Sort by energy
            sort!((a, b) => a.energy < b.energy)(allStates);
            
            // Create new temperature assignments
            auto newTemps = new double[_numReplicas];
            foreach (i; 0.._numReplicas) {
                newTemps[allStates[i].stateIndex] = temperatures[i];
            }
            
            return newTemps;
        }
        
    public:
        this(double tolerance = 1e-6, size_t maxIterations = 1000,
             size_t numReplicas = 4, size_t numProcesses = 0,
             double minTemperature = 0.1, double maxTemperature = 2.0,
             double proposalStepSize = 1e-6) {
            super(tolerance, maxIterations);
            _numReplicas = numReplicas;
            _numProcesses = numProcesses > 0 ? numProcesses : totalCPUs;
            _minTemperature = minTemperature;
            _maxTemperature = maxTemperature;
            _proposalStepSize = proposalStepSize;
        }
        
        // Handle worker process logic - now includes proposal generation and acceptance
        private void handleWorkerProcess(
            ObjectiveFunction!(T, V) objective,
            string inputQueueName,
            string resultsQueueName
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
                        proposedState[i] += normal(0, _proposalStepSize);
                    }
                    
                    // Evaluate proposal
                    double proposedEnergy = objective.evaluate(proposedState);
                    
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
        
        override OptimizationResult!(T, V) minimize(
            OptimizationState!V initialState,
            ObjectiveFunction!(T, V) objective
        ) {
            // Initialize queue manager
            auto queueManager = new TemperingQueueManager!V();
            scope(exit) queueManager.cleanup();

            // Create worker process manager
            _processManager = new ProcessManager(_numProcesses);
            scope(exit) _processManager.cleanupProcesses();

            // Setup queues
            queueManager.initialize();
            
            // Initialize temperatures and states
            auto replicaTemperatures = initializeTemperatures();
            auto replicaConfigurations = new ReplicaConfiguration!V[_numReplicas];
            auto replicaEnergies = new double[_numReplicas];
            
            // Initialize all replicas
            foreach (i; 0.._numReplicas) {
                replicaConfigurations[i] = initialState.dup;
                replicaEnergies[i] = objective.evaluate(replicaConfigurations[i]);
            }

            // Spawn workers
            _processManager.spawnWorkers((size_t workerId) {
                handleWorkerProcess(
                    objective,
                    queueManager.getInputQueueName(),
                    queueManager.getResultsQueueName()
                );
            });
            
            // Track best solution
            auto best = BestSolution!V(initialState.dup, double.infinity);
            
            // Main optimization loop
            for (size_t iter = 0; iter < _maxIterations; ++iter) {
                // Send current state to workers
                foreach (i; 0.._numReplicas) {
                    auto input = WorkerInput!V(
                        replicaConfigurations[i],
                        replicaEnergies[i],
                        replicaTemperatures[i]
                    );
                    input.send(queueManager.getInputQueue());
                }

                // Collect results
                foreach (i; 0.._numReplicas) {
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
            
            return OptimizationResult!(T, V)(best.state.positions, best.state.multiplier);
        }
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
class GradientDescentSolver(T, V) : OptimizationSolver!(T, V) {
    private:
        double _learningRate = 0.01;
        double _stepSize = 0.01;
        double _initialStep = 0.01;
        double _minStep = 1e-10;
        double _maxStep = 1.0;
        double _momentum = 0.9;
        GradientUpdateMode _updateMode = GradientUpdateMode.LearningRate;
        double _finiteDifferenceStep = 1e-6;
        size_t _numWorkers;  // Number of worker processes for gradient calculation

        struct StateWithVelocity {
            OptimizationState!V state;
            OptimizationState!V velocity;
            
            this(size_t numPositions) {
                state = OptimizationState!V.zero(numPositions);
                velocity = OptimizationState!V.zero(numPositions);
            }
        }

        // Calculate numerical gradients in parallel
        OptimizationState!V calculateGradient(
            const StateWithVelocity current,
            ObjectiveFunction!(T, V) objective
        ) {
            const size_t numPoints = current.state.positions.length;
            auto result = OptimizationState!V.zero(numPoints);

            // Create single queue for all gradient components
            string queueName = format("/gradient_%d_%s", 
                thisProcessID(), 
                randomUUID().toString()[0..8]);
            auto gradientQueue = new JumboMessageQueue(queueName);
            scope(exit) JumboMessageQueue.cleanup(queueName);

            // Create worker processes
            auto pids = new pid_t[_numWorkers];

            // Spawn worker processes
            foreach (i; 0.._numWorkers) {
                pids[i] = fork();
                if (pids[i] == 0) {
                    auto queue = new JumboMessageQueue(queueName);
                    
                    // Process each component assigned to this worker
                    for (size_t idx = i; idx < current.state.numComponents; idx += _numWorkers) {
                        // Calculate partial derivative for this component
                        auto forward_state = current.state.dup();
                        forward_state[idx] += _finiteDifferenceStep;
                        double forward = objective.evaluate(forward_state);

                        auto backward_state = current.state.dup();
                        backward_state[idx] -= _finiteDifferenceStep;
                        double backward = objective.evaluate(backward_state);

                        double derivative = (forward - backward) / (2.0 * _finiteDifferenceStep);

                        // Send result for this component
                        auto msg = GradientMessage(i, idx, derivative);
                        msg.send(queue);
                    }
                    exit(0);
                }
            }

            // Collect all component gradients from workers
            size_t totalComponents = current.state.numComponents;
            for (size_t received = 0; received < totalComponents; received++) {
                auto msg = GradientMessage.receive(gradientQueue);
                result[msg.index] = msg.value;
            }

            // Cleanup worker processes
            foreach (pid; pids) {
                if (pid != 0) {
                    wait(null);  // Wait for each process to exit
                }
            }
            
            return result;
        }

        // Calculate BB step size
        private double calculateBBStepSize(
            const OptimizationState!V gradient,
            const OptimizationState!V state,
            const OptimizationState!V previousGradient,
            const OptimizationState!V previousState,
        ) {
            auto s = state - previousState;          // Change in position
            auto y = gradient - previousGradient;    // Change in gradient
            
            double s_dot_y = 0.0;
            double y_dot_y = 0.0;
            double s_dot_s = 0.0;
            
            // Calculate dot products
            foreach (i; 0..state.numComponents) {
                s_dot_y += s[i] * y[i];
                y_dot_y += y[i] * y[i];
                s_dot_s += s[i] * s[i];
            }
            
            if (s_dot_y == 0) {
                // Avoid division by zero
                return 0;
            }

            double bb1 = s_dot_y / y_dot_y;  // BB1 step size
            double bb2 = s_dot_s / s_dot_y;  // BB2 step size
            
            // Auto mode selection based on which step gives better descent direction
            if (_updateMode == GradientUpdateMode.BBAuto) {
                // Choose BB2 if oscillating (indicated by negative s_dot_y)
                return (s_dot_y < 0) ? bb2 : bb1;
            }
            else if (_updateMode == GradientUpdateMode.BB1) {
                // Use BB1 step size
                return bb1;
            }
            else if (_updateMode == GradientUpdateMode.BB2) {
                // Use BB2 step size
                return bb2;
            }
            else {
                assert(0, "Invalid update mode");
            }
        }
        
    public:
        this(double tolerance = 1e-6, size_t maxIterations = 1000,
             double learningRate = 0.01, double stepSize = 0.01,
             GradientUpdateMode mode = GradientUpdateMode.LearningRate,
             double momentum = 0.9, double finiteDifferenceStep = 1e-6,
             double initialStep = 0.01, double minStep = 1e-10, 
             double maxStep = 1.0, size_t numWorkers = totalCPUs) {
            super(tolerance, maxIterations);
            _learningRate = learningRate;
            _stepSize = stepSize;
            _updateMode = mode;
            _momentum = momentum;
            _finiteDifferenceStep = finiteDifferenceStep;
            _initialStep = initialStep;
            _minStep = minStep;
            _maxStep = maxStep;
            _numWorkers = numWorkers;
        }
        
        override OptimizationResult!(T, V) minimize(
            OptimizationState!V initialState,
            ObjectiveFunction!(T, V) objective
        ) {
            // Initialize state with velocity
            auto state = StateWithVelocity(initialState.positions.length);
            state.state = initialState.dup;

            StateWithVelocity previousState;
            OptimizationState!V previousGradient;
            bool havePreviousState = false;
            
            // Initial evaluation
            double currentValue = objective.evaluate(state.state);
            double previousValue;
            
            // Main optimization loop
            for (size_t iter = 0; iter < _maxIterations; ++iter) {
                previousValue = currentValue;
                
                // Calculate gradients
                auto gradient = calculateGradient(state, objective);
                
                // Compute total gradient magnitude
                double totalMagnitude = sqrt(gradient.magnitudeSquared());
                
                double step;
                OptimizationState!V updateDirection;

                if (_updateMode == GradientUpdateMode.LearningRate) {
                    step = _learningRate;
                    updateDirection = gradient;
                }
                else if (_updateMode == GradientUpdateMode.StepSize) {
                    step = _stepSize;
                    updateDirection = gradient / totalMagnitude;  // Normalize
                }
                else {  // BB modes
                    if (!havePreviousState) {
                        // Use initial step size if no previous state
                        step = _initialStep;
                        havePreviousState = true;
                    } else {
                        // Calculate step size using BB method
                        step = calculateBBStepSize(gradient, state.state, previousGradient, previousState.state);
                    }
                    updateDirection = gradient;
                }
                
                debug {
                    writefln("Gradient Magnitude: %s, Step size: %s", totalMagnitude, step);
                }

                previousState = state;
                previousGradient = gradient;

                // Update velocity using momentum
                state.velocity = state.velocity * _momentum + updateDirection * (1 - _momentum);
                state.state = state.state - state.velocity * min(max(step, _minStep), _maxStep);
                
                
                // Evaluate current state
                currentValue = objective.evaluate(state.state);

                // Check convergence
                if ((abs(currentValue - previousValue) / previousValue) < _tolerance) {
                    break;
                }
            }
                
            debug {
                writefln("Best energy: %s", currentValue);
            }
            
            return OptimizationResult!(T, V)(
                state.state.positions,
                state.state.multiplier
            );
        }
}

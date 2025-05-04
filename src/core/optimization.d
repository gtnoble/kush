module core.optimization;

import core.material_point;
import math.vector;
import std.math : abs, sqrt, exp;
import std.random : uniform01;
import std.mathspecial : normalDistributionInverse;

// Helper for normal distribution sampling
private double normal(double mean, double stddev) {
    return mean + stddev * normalDistributionInverse(uniform01());
}
import core.sys.posix.unistd : fork, pid_t;
import core.sys.posix.sys.wait : wait;
import std.socket : Socket, UnixAddress, AddressFamily, SocketType;
import std.file : remove;
import std.conv : to;
import core.stdc.stdlib : exit;
import std.parallelism : totalCPUs;
import std.process : thisProcessID;
import std.uuid : randomUUID;
// Result type containing positions and scalar Lagrange multiplier
struct OptimizationResult(T, V) if (isMaterialPoint!(T, V)) {
    V[] positions;
    double multiplier;    // Single scalar multiplier for all velocity constraints
    
    this(V[] pos, double mult = 0.0) {
        positions = pos;
        multiplier = mult;
    }
}

/// Create an optimizer based on configuration
OptimizationSolver!(T, V) createOptimizer(T, V)(
    double tolerance, size_t maxIterations,
    string solverType, double learningRate,
    double stepSize, double momentum,
    string gradientMode,
    size_t numReplicas = 0,
    double minTemperature = 0.1,
    double maxTemperature = 2.0
) if (isMaterialPoint!(T, V)) {
    if (solverType == "gradient_descent") {
        auto mode = gradientMode == "learning_rate" ?
            GradientUpdateMode.LearningRate : GradientUpdateMode.StepSize;
        return new GradientDescentSolver!(T, V)(
            tolerance,
            maxIterations,
            learningRate,
            stepSize,
            mode,
            momentum
        );
    } else if (solverType == "parallel_tempering") {
        return new ParallelTemperingSolver!(T, V)(
            tolerance,
            maxIterations,
            numReplicas > 0 ? numReplicas : totalCPUs,
            minTemperature,
            maxTemperature
        );
    }
    
    throw new Exception("Unknown solver type: " ~ solverType);
}

// Interface for optimization objective functions
interface ObjectiveFunction(T, V) if (isMaterialPoint!(T, V)) {
    // Evaluate objective function with positions and scalar multiplier
    double evaluate(V[] positions, double multiplier = 0.0);
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
            V[] initialPositions,
            double initialMultiplier,
            ObjectiveFunction!(T, V) objective
        );
}

/**
 * Optimization implementations available:
 * 1. Gradient descent: Standard gradient-based optimization with momentum and learning rate/step size control
 * 2. Parallel tempering: Multi-replica optimization using MCMC sampling at different temperatures for better
 *    exploration of non-convex problems. Features:
 *    - Replicas: Multiple copies of system at different temperatures (defaults to CPU core count)
 *    - Temperature range: From min_temperature (cold/exploitation) to max_temperature (hot/exploration)
 *    - Energy-based replica swapping: Replicas exchange temperatures based on energy ordering
 *    - Metropolis acceptance: Probabilistic acceptance of proposed moves
 *
 * See OptimizationConfig in io.simulation_loader for configuration details.
 */

// Optimizer selection enum
enum OptimizerType {
    GradientDescent,
    ParallelTempering
}

/**
 * Internal messaging system for parallel tempering:
 * - Parent -> Replica: Temperature assignments for sampling
 * - Replica -> Parent: State reports containing positions and energy
 *
 * Serialization format:
 * - Temperature: Single double value
 * - State report: [energy, multiplier, positions...]
 */

// Message for temperature assignment (parent -> replica)
private struct TemperatureAssignment {
    double temperature;

    void serialize(ref ubyte[] buffer) const {
        buffer = new ubyte[double.sizeof];
        (cast(double*)buffer.ptr)[0] = temperature;
    }

    static TemperatureAssignment deserialize(const(ubyte)[] buffer) {
        return TemperatureAssignment((cast(const(double)*)buffer.ptr)[0]);
    }
}

// Message for replica state report (replica -> parent)
private struct ReplicaReport(V) {
    double energy;
    V[] positions;
    double multiplier;

    void serialize(ref ubyte[] buffer) const {
        // Calculate total size needed
        size_t messageSize = 2 * double.sizeof + positions.length * V.sizeof;
        buffer = new ubyte[messageSize];
        
        // Write energy and multiplier
        (cast(double*)buffer.ptr)[0] = energy;
        (cast(double*)(buffer.ptr + double.sizeof))[0] = multiplier;
        
        // Write positions array
        size_t posOffset = 2 * double.sizeof;
        foreach (i, pos; positions) {
            (cast(V*)(buffer.ptr + posOffset + i * V.sizeof))[0] = pos;
        }
    }

    static ReplicaReport!V deserialize(const(ubyte)[] buffer, size_t numPositions) {
        ReplicaReport!V report;
        report.energy = (cast(const(double)*)buffer.ptr)[0];
        report.multiplier = (cast(const(double)*)(buffer.ptr + double.sizeof))[0];
        
        // Deserialize positions
        report.positions = new V[numPositions];
        size_t posOffset = 2 * double.sizeof;
        foreach (i; 0..numPositions) {
            report.positions[i] = (cast(const(V)*)(buffer.ptr + posOffset + i * V.sizeof))[0];
        }
        return report;
    }
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
 * - IPC using Unix domain sockets for replica communication
 * - Adaptive step sizes based on temperature
 * - Automatic cleanup of child processes and sockets
 *
 * Configuration in JSON:
 * {
 *   "solver_type": "parallel_tempering",
 *   "tolerance": 1e-6,
 *   "max_iterations": 1000,
 *   "parallel_tempering": {
 *     "num_replicas": 8,  // Optional: defaults to CPU count
 *     "min_temperature": 0.1,
 *     "max_temperature": 2.0
 *   }
 * }
 */
class ParallelTemperingSolver(T, V) : OptimizationSolver!(T, V) {
    private:
        size_t _numReplicas;
        double _minTemperature;
        double _maxTemperature;
        double _gradientStepSize = 1e-6;

        struct ReplicaState {
            pid_t pid;
            Socket socket;
            double temperature;
            double energy;
            V[] positions;
            double multiplier;
        }
        ReplicaState[] replicas;

        // Initialize replica temperatures using geometric progression
        void initializeTemperatures() {
            import std.math : pow;
            double ratio = pow(_maxTemperature / _minTemperature, 1.0 / (_numReplicas - 1));
            foreach (i, ref replica; replicas) {
                replica.temperature = _minTemperature * pow(ratio, i);
            }
        }

        // Sort replicas by energy and reassign temperatures
        void reorderTemperatures() {
            import std.algorithm : sort;
            
            // Store original temperatures
            auto temps = new double[_numReplicas];
            foreach (i; 0.._numReplicas) {
                temps[i] = replicas[i].temperature;
            }
            
            // Sort replicas by energy
            sort!((a, b) => a.energy < b.energy)(replicas);
            
            // Reassign temperatures
            foreach (i; 0.._numReplicas) {
                replicas[i].temperature = temps[i];
            }
        }

    public:
        this(double tolerance = 1e-6, size_t maxIterations = 1000,
             size_t numReplicas = 4, double minTemperature = 0.1,
             double maxTemperature = 2.0) {
            super(tolerance, maxIterations);
            _numReplicas = numReplicas;
            _minTemperature = minTemperature;
            _maxTemperature = maxTemperature;
        }

        override OptimizationResult!(T, V) minimize(
            V[] initialPositions,
            double initialMultiplier,
            ObjectiveFunction!(T, V) objective
        ) {
            // Initialize replicas
            replicas = new ReplicaState[_numReplicas];
            string[] socketPaths;
            
            // Create replicas
            foreach (i; 0.._numReplicas) {
                // Setup socket
                string socketPath = "/tmp/tempering_" ~ thisProcessID().to!string ~ "_" ~
                    randomUUID().toString()[0..8] ~ "_" ~ i.to!string;
                socketPaths ~= socketPath;
                
                auto serverSocket = new Socket(AddressFamily.UNIX, SocketType.STREAM);
                scope(exit) serverSocket.close();
                serverSocket.bind(new UnixAddress(socketPath));
                serverSocket.listen(1);

                // Fork replica process
                replicas[i].pid = fork();
                if (replicas[i].pid == 0) {  // Child process
                    serverSocket.close();
                    
                    // Connect to parent
                    auto socket = new Socket(AddressFamily.UNIX, SocketType.STREAM);
                    scope(exit) socket.close();
                    socket.connect(new UnixAddress(socketPath));
                    
                    // Save initial positions
                    auto savedPositions = initialPositions.dup;
                    double savedMultiplier = initialMultiplier;
                    
                    // Initialize replica state
                    auto state = ReplicaState(0, null, 0.0, 
                        objective.evaluate(savedPositions, savedMultiplier),
                        savedPositions, savedMultiplier);

                    // Replica process main loop
                    while (true) {
                        // Receive temperature assignment
                        ubyte[] buffer = new ubyte[double.sizeof];
                        socket.receive(buffer);
                        auto assignment = TemperatureAssignment.deserialize(buffer);
                        state.temperature = assignment.temperature;

                        // Generate proposal using random walk
                        foreach (ref pos; state.positions) {
                            foreach (ref component; pos.components) {
                                component += normal(0, sqrt(state.temperature)) * _gradientStepSize;
                            }
                        }
                        state.multiplier += normal(0, sqrt(state.temperature)) * _gradientStepSize;

                        // Evaluate proposal
                        double proposalEnergy = objective.evaluate(state.positions, state.multiplier);

                        // Metropolis acceptance
                        // Store current state before proposal
                        auto currentPositions = state.positions.dup;
                        double currentMultiplier = state.multiplier;
                        double currentEnergy = state.energy;

                        // Check acceptance
                        if (proposalEnergy <= currentEnergy || 
                            uniform01() < exp((currentEnergy - proposalEnergy) / state.temperature)) {
                            state.energy = proposalEnergy;
                        } else {
                            // Reject proposal, revert state
                            state.positions[] = currentPositions[];
                            state.multiplier = currentMultiplier;
                            state.energy = currentEnergy;
                        }

                        // Send state report
                        auto report = ReplicaReport!V(
                            state.energy,
                            state.positions,
                            state.multiplier
                        );
                        report.serialize(buffer);
                        socket.send(buffer);
                    }
                }

                // Parent process accepts connection
                replicas[i].socket = serverSocket.accept();
            }

            // Initialize temperatures
            initializeTemperatures();

            // Track best solution
            double bestEnergy = double.infinity;
            size_t bestIndex;

            // Main optimization loop
            for (size_t iter = 0; iter < _maxIterations; ++iter) {
                // Assign temperatures and collect results
                foreach (i, ref replica; replicas) {
                    // Send temperature assignment
                    auto assignment = TemperatureAssignment(replica.temperature);
                    ubyte[] buffer;
                    assignment.serialize(buffer);
                    replica.socket.send(buffer);

                    // Receive state report
                    buffer = new ubyte[2 * double.sizeof + initialPositions.length * V.sizeof];
                    replica.socket.receive(buffer);
                    auto report = ReplicaReport!V.deserialize(buffer, initialPositions.length);

                    // Update replica state
                    replica.energy = report.energy;
                    replica.positions = report.positions;
                    replica.multiplier = report.multiplier;

                    // Track best solution
                    if (replica.energy < bestEnergy) {
                        bestEnergy = replica.energy;
                        bestIndex = i;
                    }
                }

                // Reorder temperatures based on energies
                reorderTemperatures();

                // Check convergence
                if (iter > 0 && abs(replicas[0].energy - bestEnergy) < _tolerance) {
                    break;
                }
            }

            // Cleanup
            foreach (replica; replicas) {
                if (replica.socket !is null) replica.socket.close();
            }
            foreach (path; socketPaths) {
                remove(path);
            }
            foreach (replica; replicas) {
                int status;
                wait(&status);
            }

            return OptimizationResult!(T, V)(
                replicas[bestIndex].positions.dup,
                replicas[bestIndex].multiplier
            );
        }
}

/**
 * Update modes for gradient descent optimization:
 * - LearningRate: Scale gradient by learning rate (traditional gradient descent)
 * - StepSize: Move fixed distance in gradient direction (normalized gradient)
 */
enum GradientUpdateMode {
    LearningRate,  // Scale gradient by learning rate
    StepSize       // Move fixed step size in gradient direction
}

// Gradient message for parallel computation
private struct GradientMessage(V) {
    size_t index;
    V positionGradient;
    double multiplierGradient;  // Now scalar

    void serialize(ref ubyte[] buffer) const {
        buffer = new ubyte[this.sizeof];
        (cast(GradientMessage!V*)buffer.ptr)[0] = this;
    }

    static GradientMessage!V deserialize(const(ubyte)[] buffer) {
        return *(cast(const(GradientMessage!V)*)buffer.ptr);
    }
}

/**
 * Gradient descent solver with momentum and parallel gradient computation.
 *
 * Features:
 * - Momentum-based updates for faster convergence
 * - Parallel gradient computation across multiple processes
 * - Support for both learning rate and step size modes
 * - Automatic gradient normalization
 * 
 * Configuration in JSON:
 * {
 *   "solver_type": "gradient_descent",
 *   "tolerance": 1e-6,
 *   "max_iterations": 1000,
 *   "momentum": 0.9,
 *   "gradient_mode": "step_size",  // or "learning_rate"
 *   "learning_rate": 0.01,  // Used if mode is "learning_rate"
 *   "gradient_step_size": {
 *     "value": 1e-4,
 *     "horizon_fraction": 0.0001  // Optional: step = horizon * fraction
 *   }
 * }
 */
class GradientDescentSolver(T, V) : OptimizationSolver!(T, V) {
    private:
        double _learningRate = 0.01;
        double _stepSize = 0.01;
        double _momentum = 0.9;
        GradientUpdateMode _updateMode = GradientUpdateMode.LearningRate;
        double _gradientStepSize = 1e-6;
        size_t _numWorkers;  // Number of worker processes for gradient calculation

        struct OptimizationState {
            V[] positions;
            V[] positionVelocities;
            double multiplier;           // Single scalar
            double multiplierVelocity;   // Single scalar velocity
        }

        struct GradientResult {
            V[] positionGradients;
            double multiplierGradient;   // Single scalar gradient
        }

        // Calculate numerical gradients in parallel
        GradientResult calculateGradient(
            OptimizationState current,
            ObjectiveFunction!(T, V) objective
        ) {
            const size_t numPoints = current.positions.length;
            auto result = GradientResult(
                new V[numPoints],
                0.0  // Initialize scalar multiplier gradient
            );

            auto sockets = new Socket[_numWorkers];
            auto pids = new pid_t[_numWorkers];
            string[] socketPaths;
            
            // Setup sockets for parallel computation
            foreach (i; 0.._numWorkers) {
                // Create unique socket path with process ID and random component
                string socketPath = "/tmp/gradient_" ~ thisProcessID().to!string ~ "_" ~ 
                    randomUUID().toString()[0..8] ~ "_" ~ i.to!string;
                socketPaths ~= socketPath;
                auto serverSocket = new Socket(AddressFamily.UNIX, SocketType.STREAM);
                scope(exit) serverSocket.close();
                serverSocket.bind(new UnixAddress(socketPath));
                serverSocket.listen(1);
                
                pids[i] = fork();
                if (pids[i] == 0) {  // Child process
                    serverSocket.close();
                    
                    auto socket = new Socket(AddressFamily.UNIX, SocketType.STREAM);
                    scope(exit) socket.close();
                    socket.connect(new UnixAddress(socketPath));
                    
                    // Process assigned points in parallel
                    for (size_t j = i; j < numPoints; j += _numWorkers) {
                        V posGradient = V.zero();
                        
                        // Calculate position gradients
                        foreach (dim; 0..V.dimension) {
                            // Forward difference for position
                            {
                                auto pos = current.positions.dup;
                                pos[j][dim] += _gradientStepSize;
                                double forward = objective.evaluate(pos, current.multiplier);

                                // Backward difference for position
                                pos[j][dim] -= 2 * _gradientStepSize;
                                double backward = objective.evaluate(pos, current.multiplier);

                                // Central difference
                                posGradient[dim] = 
                                    (forward - backward) / (2.0 * _gradientStepSize);
                            }
                        }
                        
                        // Send result to parent
                        // Note: multiplier gradient is handled separately
                        auto msg = GradientMessage!V(j, posGradient, 0.0);
                        ubyte[] buffer;
                        msg.serialize(buffer);
                        socket.send(buffer);
                    }
                    
                    exit(0);
                }
                
                // Parent process accepts connection
                sockets[i] = serverSocket.accept();
            }
            
            // Collect results from workers
            ubyte[] buffer = new ubyte[GradientMessage!V.sizeof];
            foreach (i; 0..numPoints) {
                auto bytesReceived = sockets[i % _numWorkers].receive(buffer);
                if (bytesReceived == GradientMessage!V.sizeof) {
                    auto msg = GradientMessage!V.deserialize(buffer);
                    result.positionGradients[msg.index] = msg.positionGradient;
                }
            }
            
            // Cleanup
            scope(exit) {
                foreach (socket; sockets) {
                    if (socket !is null) socket.close();
                }
                foreach (path; socketPaths) {
                    remove(path);
                }
            }
            
            // Wait for child processes
            foreach (i; 0.._numWorkers) {
                int status;
                wait(&status);
            }

            // Calculate multiplier gradient (done in parent process)
            {
                double forward = objective.evaluate(
                    current.positions,
                    current.multiplier + _gradientStepSize
                );
                double backward = objective.evaluate(
                    current.positions,
                    current.multiplier - _gradientStepSize
                );
                result.multiplierGradient = (forward - backward) / (2.0 * _gradientStepSize);
            }
            
            return result;
        }

        // Update state using momentum
        void updateState(
            ref OptimizationState state,
            GradientResult gradient
        ) {
            // Compute total gradient magnitude including position gradients and scalar multiplier gradient
            double totalMagnitudeSquared = 0.0;
            foreach (i; 0..state.positions.length) {
                totalMagnitudeSquared += gradient.positionGradients[i].magnitudeSquared();
            }
            totalMagnitudeSquared += gradient.multiplierGradient * gradient.multiplierGradient;
            double totalMagnitude = sqrt(totalMagnitudeSquared);
            
            // Skip update if gradient is too small
            if (totalMagnitude < 1e-10) return;
            
            // Update positions
            for (size_t i = 0; i < state.positions.length; ++i) {
                final switch (_updateMode) {
                    case GradientUpdateMode.LearningRate:
                        state.positionVelocities[i] = 
                            state.positionVelocities[i] * _momentum - 
                            (gradient.positionGradients[i] / totalMagnitude) * _learningRate;
                        break;

                    case GradientUpdateMode.StepSize:
                        state.positionVelocities[i] = 
                            state.positionVelocities[i] * _momentum - 
                            (gradient.positionGradients[i] / totalMagnitude) * _stepSize;
                        break;
                }
                state.positions[i] = state.positions[i] + state.positionVelocities[i];
            }
            
            // Update multiplier
            final switch (_updateMode) {
                case GradientUpdateMode.LearningRate:
                    state.multiplierVelocity = 
                        state.multiplierVelocity * _momentum - 
                        (gradient.multiplierGradient / totalMagnitude) * _learningRate;
                    break;

                case GradientUpdateMode.StepSize:
                    state.multiplierVelocity = 
                        state.multiplierVelocity * _momentum - 
                        (gradient.multiplierGradient / totalMagnitude) * _stepSize;
                    break;
            }
            state.multiplier = state.multiplier + state.multiplierVelocity;
        }
        
    public:
        this(double tolerance = 1e-6, size_t maxIterations = 1000,
             double learningRate = 0.01, double stepSize = 0.01,
             GradientUpdateMode mode = GradientUpdateMode.LearningRate,
             double momentum = 0.9, size_t numWorkers = totalCPUs) {
            super(tolerance, maxIterations);
            _learningRate = learningRate;
            _stepSize = stepSize;
            _updateMode = mode;
            _momentum = momentum;
            _numWorkers = numWorkers;
        }
        
        override OptimizationResult!(T, V) minimize(
            V[] initialPositions,
            double initialMultiplier,
            ObjectiveFunction!(T, V) objective
        ) {
            // Initialize optimization state
            auto state = OptimizationState(
                initialPositions.dup,
                new V[initialPositions.length],  // Zero velocities for positions
                initialMultiplier,
                0.0  // Zero velocity for multiplier
            );
            
            // Set initial position velocities to zero
            foreach (ref v; state.positionVelocities) v = V.zero();
            
            // Initial evaluation
            double currentValue = objective.evaluate(state.positions, state.multiplier);
            double previousValue;
            
            // Main optimization loop
            for (size_t iter = 0; iter < _maxIterations; ++iter) {
                previousValue = currentValue;
                
                // Calculate gradients
                auto gradients = calculateGradient(state, objective);
                
                // Update state using momentum
                updateState(state, gradients);
                
                // Evaluate current state
                currentValue = objective.evaluate(state.positions, state.multiplier);
                
                // Check convergence
                if (abs(currentValue - previousValue) < _tolerance) {
                    break;
                }
            }
            
            return OptimizationResult!(T, V)(state.positions, state.multiplier);
        }
}

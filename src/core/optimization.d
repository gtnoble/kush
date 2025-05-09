module core.optimization;

import core.material_point;
import math.vector;
import std.math : abs, sqrt, exp;
import std.algorithm.comparison : min;
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
import std.socket : Socket, SocketOptionLevel, SocketOption, AddressFamily, SocketType, UnixAddress;
import std.uuid : randomUUID;
import std.file : remove;
import core.sys.posix.poll : poll, pollfd, POLLIN;

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

private struct ReplicaConfiguration(V) {
    V[] positions;  // Current positions of the replica
    double multiplier;  // Scalar multiplier for velocity constraints

    void send(Socket socket) {
        socket.send((&multiplier)[0..1]);

        // Send positions array size and data
        size_t size = positions.length;
        socket.send((&size)[0..1]);
        socket.send(positions);
    }
        

    static receiveJob(Socket socket) {
        ReplicaConfiguration!V job;
        
        // Receive parameters
        socket.receive((&job.multiplier)[0..1]);
        
        // Receive positions array size and data
        size_t size;
        socket.receive((&size)[0..1]);
        job.positions = new V[size];
        auto received = socket.receive(job.positions);
        
        if (received != V.sizeof * size) {
            throw new Exception("Incomplete position data received");
        }
        
        return job;
    }
}

/**
 * Socket manager for worker communication.
 * Handles creation, connection, and cleanup of Unix domain sockets.
 */
// Worker jobs and state
class WorkerSocketManager(V) {
    private:
        Socket[] _serverSockets;    // Listening sockets
        Socket[] _connections;      // Active connections
        string[] _socketPaths;
        size_t _numWorkers;
        bool _initialized;
        pollfd[] _pollFds;         // For poll-based monitoring
        
    public:
        this(size_t numWorkers) {
            _numWorkers = numWorkers;
            _serverSockets = new Socket[numWorkers];
            _connections = new Socket[numWorkers];
            _socketPaths = new string[numWorkers];
            _initialized = false;
        }
        
        void initializeSockets() {
            if (_initialized) return;
            
            foreach (i; 0.._numWorkers) {
                // Create unique socket path
                _socketPaths[i] = format("/tmp/pd_worker_%d_%s_%d",
                    thisProcessID(),
                    randomUUID().toString()[0..8],
                    i);
                
                // Create and bind socket
                _serverSockets[i] = new Socket(AddressFamily.UNIX, SocketType.STREAM);
                _serverSockets[i].bind(new UnixAddress(_socketPaths[i]));
                _serverSockets[i].listen(1);
            }
            
            _initialized = true;
        }
        
        void acceptConnections() {
            import std.datetime: dur, Duration;
            
            foreach (i; 0.._numWorkers) {
                // Set accept timeout
                _serverSockets[i].setOption(SocketOptionLevel.SOCKET,
                                          SocketOption.RCVTIMEO,
                                          dur!"seconds"(5));
                                          
                // Accept connection
                try {
                    _connections[i] = _serverSockets[i].accept();
                } catch (Exception e) {
                    throw new Exception(format(
                        "Worker %d failed to connect within timeout: %s", 
                        i, e.msg));
                }
                
                // Close server socket, no longer needed
                _serverSockets[i].close();
                _serverSockets[i] = null;
            }
            
            import core.sys.posix.poll : POLLIN, POLLOUT;
            
            // Initialize poll structures
            _pollFds = new pollfd[_connections.length];
            foreach (i; 0.._connections.length) {
                _pollFds[i] = pollfd(
                    _connections[i].handle,  // fd
                    POLLIN | POLLOUT,       // events to monitor (read and write)
                    0                        // returned events
                );
            }
        }
        
        ~this() {
            // Close active connections
            foreach (socket; _connections) {
                if (socket !is null) socket.close();
            }
            
            // Close any remaining server sockets
            foreach (socket; _serverSockets) {
                if (socket !is null) socket.close();
            }
            
            // Clean up socket files
            foreach (path; _socketPaths) {
                remove(path);
            }
        }
        
        // Returns array of worker indices that have events (read or write)
        ReadyStatus[] checkReadyWorkers(int timeout = 0) {
            import core.sys.posix.poll : POLLIN, POLLOUT;
            
            ReadyStatus[] readyWorkers = new ReadyStatus[_numWorkers];
            
            // Poll for available data or write availability
            auto result = poll(_pollFds.ptr, _pollFds.length, timeout);
            if (result < 0) {
                import core.stdc.errno : errno;
                throw new Exception(
                    "Poll failed: " ~ 
                    fromStringz(strerror(errno)).idup
                );
            }
            
            if (result > 0) {
                foreach (i; 0.._pollFds.length) {
                    if (_pollFds[i].revents & POLLIN & POLLOUT) {
                        readyWorkers[i] = ReadyStatus.ReadyToReadWrite;
                    }
                    else if (_pollFds[i].revents & POLLIN) {
                        readyWorkers[i] = ReadyStatus.ReadyToRead;
                    }
                    else if (_pollFds[i].revents & POLLOUT) {
                        readyWorkers[i] = ReadyStatus.ReadyToWrite;
                    }
                    else {
                        readyWorkers[i] = ReadyStatus.NotReady;
                    }
                }
            }
            
            return readyWorkers;
        }
        
        double receiveEnergy(size_t workerId) {
            double energy;
            _connections[workerId].receive((&energy)[0..1]);
            return energy;
        }
        
        string getSocketPath(size_t workerId) {
            return _socketPaths[workerId];
        }
        
        Socket getSocket(size_t workerId) {
            return _connections[workerId];
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
        auto mode = config.gradient_mode == "learning_rate" ?
            GradientUpdateMode.LearningRate : GradientUpdateMode.StepSize;
        return new GradientDescentSolver!(T, V)(
            config.tolerance,
            config.max_iterations,
            config.learning_rate,
            config.getEffectiveStepSize(horizon),
            mode,
            config.momentum
        );
    } else if (config.solver_type == "parallel_tempering") {
        return new ParallelTemperingSolver!(T, V)(
            config.tolerance,
            config.max_iterations,
            config.getNumReplicas(),
            config.getNumProcesses(),
            config.parallel_tempering.min_temperature,
            config.parallel_tempering.max_temperature
        );
    }
    
    throw new Exception("Unknown solver type: " ~ config.solver_type);
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
 *     "min_temperature": 0.1,
 *     "max_temperature": 2.0
 *   }
 * }
 */
class ParallelTemperingSolver(T, V) : OptimizationSolver!(T, V) {
    private:
        static struct BestSolution(V) {
            V[] positions;
            double multiplier;
            double energy = double.infinity;

            this(V[] pos, double mult, double e = double.infinity) {
                positions = pos.dup;
                multiplier = mult;
                energy = e;
            }
        }

        size_t _numReplicas;
        size_t _numProcesses;
        double _minTemperature;
        double _maxTemperature;
        double _gradientStepSize = 1e-6;
        
        // Resource management
        ProcessManager _processManager;
        WorkerSocketManager!V _socketManager;
        
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
             double minTemperature = 0.1, double maxTemperature = 2.0) {
            super(tolerance, maxIterations);
            _numReplicas = numReplicas;
            _numProcesses = numProcesses > 0 ? numProcesses : totalCPUs;
            _minTemperature = minTemperature;
            _maxTemperature = maxTemperature;
        }
        
        // Handle worker process logic
        private void handleWorkerProcess(
            size_t workerId,
            ObjectiveFunction!(T, V) objective
        ) {
            try {
                // Connect to the socket
                auto socket = new Socket(AddressFamily.UNIX, SocketType.STREAM);
                socket.connect(new UnixAddress(_socketManager.getSocketPath(workerId)));
                
                // Process evaluation requests
                while (true) {
                    // Read parameters
                    ReplicaConfiguration!V job = ReplicaConfiguration!V.receiveJob(socket);

                    // Evaluate objective and return result
                    double energy = objective.evaluate(job.positions, job.multiplier);
                    socket.send((&energy)[0..1]);
                }
            } catch (Exception e) {
                stderr.writefln("Worker %d error: %s", workerId, e.msg);
            }
        }
        
        // Generate proposal for replica
        ReplicaConfiguration!V generateProposal(ReplicaConfiguration!V previousReplica) {
            
            // Generate proposed positions with random walk
            auto proposedPositions = previousReplica.positions.dup;
            foreach (i, pos; proposedPositions) {
                foreach (dim; 0..V.dimension) {
                    proposedPositions[i][dim] += normal(0, _gradientStepSize);
                }
            }
            
            // Generate proposed multiplier
            double proposedMultiplier = previousReplica.multiplier + 
                normal(0, _gradientStepSize);
                
            return ReplicaConfiguration!V(
                proposedPositions,
                proposedMultiplier,
            );
        }
        
        override OptimizationResult!(T, V) minimize(
            V[] initialPositions,
            double initialMultiplier,
            ObjectiveFunction!(T, V) objective
        ) {
            // Initialize managers
            _socketManager = new WorkerSocketManager!V(_numProcesses);
            scope(exit) destroy(_socketManager);
            _processManager = new ProcessManager(_numProcesses);
            scope(exit) _processManager.cleanupProcesses();
            
            // Setup sockets first
            _socketManager.initializeSockets();
            
            // Initialize temperatures
            auto replicaTemperatures = initializeTemperatures();
            auto replicaEnergies = new double[_numReplicas];
            replicaEnergies[] = double.infinity;
            
            auto initialConfig = ReplicaConfiguration!V(
                initialPositions.dup,
                initialMultiplier
            );
            auto replicaConfigurations = new ReplicaConfiguration!V[_numReplicas];
            replicaConfigurations[] = initialConfig;

            // Spawn workers and establish connections
            _processManager.spawnWorkers((size_t workerId) {
                handleWorkerProcess(workerId, objective);
            });
            
            // Accept connections from all workers
            _socketManager.acceptConnections();
            
            // Track best solution
            auto best = BestSolution!V(
                initialPositions.dup,
                initialMultiplier,
                double.infinity
            );
            
            // Main optimization loop with asynchronous processing
            for (size_t iter = 0; iter < _maxIterations; ++iter) {
                
                int numGatheredResults = 0;
                int numJobsAssigned = 0;
                ReplicaConfiguration!V[] proposedStates = new ReplicaConfiguration!V[_numReplicas];
                double[] proposedEnergies = new double[_numReplicas];

                // Process jobs asynchronously using poll for both read and write
                while (numJobsAssigned < _numReplicas || numGatheredResults < _numReplicas) {
                    
                    // Check socket states
                    auto readyWorkers = _socketManager.checkReadyWorkers(100);  // 100ms timeout
                    
                    // Process ready sockets
                    foreach (workerId, status; readyWorkers) {
                        
                        // Handle write events (assign new jobs)
                        if (status == ReadyStatus.ReadyToWrite || status == ReadyStatus.ReadyToReadWrite) {
                            proposedStates[workerId] = generateProposal(replicaConfigurations[workerId]);
                            auto socket = _socketManager.getSocket(workerId);
                            proposedStates[workerId].send(socket);
                            numJobsAssigned++;
                        }
                        
                        // Handle read events (process completed jobs)
                        if (status == ReadyStatus.ReadyToRead || status == ReadyStatus.ReadyToReadWrite) {
                            proposedEnergies[workerId] = _socketManager.receiveEnergy(workerId);
                            numGatheredResults++;
                        }
                    }
                }

                for (size_t i = 0; i < _numReplicas; i++) {
                    // Metropolis acceptance
                    auto proposedEnergy = proposedEnergies[i];
                    auto proposedConfig = proposedStates[i];
                    if (proposedEnergy <= replicaEnergies[i] || 
                        uniform01() < exp((replicaEnergies[i] - proposedEnergy) / replicaTemperatures[i])) {
                        // Accept proposal
                        replicaConfigurations[i] = proposedConfig;
                        replicaEnergies[i] = proposedEnergy;
                        
                        // Update best solution if needed
                        if (proposedEnergy < best.energy) {
                            best.energy = proposedEnergy;
                            best.positions = proposedConfig.positions;
                            best.multiplier = proposedConfig.multiplier;
                        }
                    }

                }
                
                // Reorder temperatures based on energies
                replicaTemperatures = reorderTemperatures(replicaTemperatures, replicaEnergies);
            }
            
            return OptimizationResult!(T, V)(best.positions, best.multiplier);
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
                        auto sent = socket.send(buffer);
                        if (sent != buffer.length) {
                            throw new Exception("Failed to send complete gradient message");
                        }
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

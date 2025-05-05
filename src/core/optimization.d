module core.optimization;

import core.material_point;
import math.vector;
import std.math : abs, sqrt, exp;
import std.array;
import std.random : uniform01;
import std.mathspecial : normalDistributionInverse;
import std.stdio;
import core.sys.posix.unistd : fork, pid_t, close;
import core.sys.posix.signal : 
    kill, sigaction, sigaction_t,
    SIGTERM, SIGABRT, SIGINT, SA_RESTART;
import core.sys.posix.sys.wait : wait, WIFEXITED, WIFSIGNALED, WEXITSTATUS, WTERMSIG;
import core.sys.posix.mqueue;  // POSIX message queues
import core.sys.posix.fcntl : O_CREAT, O_RDWR, S_IRUSR, S_IWUSR;
import core.stdc.stdlib : exit;
import std.conv : to;
import std.file : readText;
import std.string : strip;
import core.sys.posix.sys.types : ssize_t;
import std.format : format;
import core.stdc.errno;
import core.stdc.string : strerror;
import std.parallelism : totalCPUs;
import std.process : thisProcessID;
import std.string : toStringz, fromStringz;
import std.socket : Socket, AddressFamily, SocketType, UnixAddress;
import std.uuid : randomUUID;
import std.file : remove;
import core.stdc.stdio : snprintf;

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
        
        extern(C) static void signalHandler(int sig) nothrow @nogc {
            import core.sys.posix.unistd : write, STDERR_FILENO;
            import core.stdc.string : strlen;

            const(char)* msg = "Process terminated by signal\n";
            write(STDERR_FILENO, msg, strlen(msg));
            exit(0);
        }

        void spawnWorkers(void delegate(size_t) worker) {
            // Set up signal handler for children
            import core.sys.posix.signal : sigaction_t, SA_RESTART, SIGTERM, SIGABRT, SIGINT;
            sigaction_t action;
            action.sa_handler = cast(void function(int))&signalHandler;
            action.sa_flags = SA_RESTART;  // Restart interrupted system calls

            // Install signal handlers
            import core.sys.posix.signal : sigaction;
            if (sigaction(SIGTERM, &action, null) == -1 ||
                sigaction(SIGABRT, &action, null) == -1 ||
                sigaction(SIGINT, &action, null) == -1) {
                throw new Exception("Failed to install signal handlers");
            }

            foreach (i; 0.._numProcesses) {
                _pids[i] = fork();
                if (_pids[i] == 0) {  // Child process
                    writeln("Child process ", i, " (PID ", thisProcessID(), ") starting");
                    try {
                        worker(i);
                    } catch (Exception e) {
                        stderr.writeln("Child process ", i, " error: ", e.msg);
                    }
                    writeln("Child process ", i, " (PID ", thisProcessID(), ") exiting");
                    exit(0);  // Exit child process
                }
                
                writeln("Parent started child ", i, " with PID ", _pids[i]);
                stdout.flush();
            }
        }
        
        void sendSignalToAll(int signal) {
            foreach (pid; _pids) {
                if (pid != 0) {
                    kill(pid, signal);
                }
            }
        }
        
        void cleanupProcesses() {
            writeln("Starting cleanup process...");
            stdout.flush();
            
            // First, log active PIDs
            writeln("Active PIDs: ", _pids);
            stdout.flush();
            
            // Send SIGTERM to all replicas
            writeln("Sending SIGTERM to all processes...");
            sendSignalToAll(SIGTERM);
            stdout.flush();
            
            foreach (i, pid; _pids) {
                if (pid != 0) {
                    writefln("Waiting for process %d (index %d)...", pid, i);
                    stdout.flush();
                    
                    int status;
                    auto result = wait(&status);
                    
                    if (result == -1) {
                        stderr.writefln("Failed to wait for process %d: %s (errno: %d)", 
                            pid, fromStringz(strerror(errno)), errno);
                        stderr.flush();
                    } else if (result != pid) {
                        stderr.writefln("Wait returned for wrong process. Expected %d, got %d",
                            pid, result);
                        stderr.flush();
                    } else if (WIFEXITED(status)) {
                        auto exitStatus = WEXITSTATUS(status);
                        writefln("Process %d exited with status %d", pid, exitStatus);
                        if (exitStatus != 0) {
                            stderr.writefln("WARNING: Process %d had non-zero exit status %d", 
                                pid, exitStatus);
                        }
                    } else if (WIFSIGNALED(status)) {
                        auto signal = WTERMSIG(status);
                        stderr.writefln("Process %d terminated by signal %d", pid, signal);
                    } else {
                        stderr.writefln("Process %d exited abnormally (status: %d)", pid, status);
                    }
                    stdout.flush();
                    stderr.flush();
                }
            }
            
            writeln("Cleanup process complete.");
            stdout.flush();
        }
}

// Message types for parallel tempering communication
enum ControlMessageType {
    TEMPERATURE_ASSIGNMENT  // No more terminate message - using signals instead
}

// Parent -> Replica control message
struct ControlMessage(V) {
    ControlMessageType type;
    double temperature;
    size_t replicaId;  // Target replica

    void serialize(ref ubyte[] buffer) const {
        buffer = new ubyte[this.sizeof];
        (cast(ControlMessage!V*)buffer.ptr)[0] = this;
    }

    static ControlMessage!V deserialize(const(ubyte)[] buffer) {
        return *(cast(const(ControlMessage!V)*)buffer.ptr);
    }
}

// Replica -> Parent state message
struct StateMessage(V) {
    size_t replicaId;
    double energy;
    double multiplier;
    V[] positions;

    void serialize(ref ubyte[] buffer) {
        // Calculate total size needed
        size_t totalSize = 
            size_t.sizeof +      // replicaId
            double.sizeof * 2 +  // energy + multiplier
            V.sizeof * positions.length + // positions array
            size_t.sizeof;       // position array length

        buffer = new ubyte[totalSize];
        size_t offset = 0;

        // Write fields
        (cast(size_t*)(buffer.ptr + offset))[0] = replicaId;
        offset += size_t.sizeof;
        
        (cast(double*)(buffer.ptr + offset))[0] = energy;
        offset += double.sizeof;
        
        (cast(double*)(buffer.ptr + offset))[0] = multiplier;
        offset += double.sizeof;

        // Write positions array length
        (cast(size_t*)(buffer.ptr + offset))[0] = positions.length;
        offset += size_t.sizeof;
        
        // Write positions array
        V* posPtr = cast(V*)(buffer.ptr + offset);
        posPtr[0..positions.length] = positions[];
    }

    static StateMessage!V deserialize(const(ubyte)[] buffer) {
        StateMessage!V msg;
        size_t offset = 0;

        // Read fields
        msg.replicaId = (cast(size_t*)(buffer.ptr + offset))[0];
        offset += size_t.sizeof;
        
        msg.energy = (cast(double*)(buffer.ptr + offset))[0];
        offset += double.sizeof;
        
        msg.multiplier = (cast(double*)(buffer.ptr + offset))[0];
        offset += double.sizeof;

        // Read positions array length
        size_t numPositions = (cast(size_t*)(buffer.ptr + offset))[0];
        offset += size_t.sizeof;
        
        // Read positions array
        msg.positions = (cast(V*)(buffer.ptr + offset))[0..numPositions].dup;
        
        return msg;
    }
}

/**
 * Message queue manager for parallel tempering communication.
 * Provides two queues:
 * 1. Control queue (parent -> replicas)
 * 2. Results queue (replicas -> parent)
 */
class MessageQueueManager(V) {
    private:
        mqd_t[] _controlQueues;  // Array of control queues (one per worker)
        mqd_t _resultQueue;      // Workers -> Parent
        size_t _msgSize;         // Size required for messages
        size_t _numReplicas;     // Number of replicas/workers
        bool _initialized;

        // Static string and buffer configuration
        static immutable QueuePrefix = "/pd_control_";
        static immutable MaxQueueDigits = 20;  // Large enough for size_t.max
        
        // Fixed-size buffers for paths with static lifetime
        static immutable ResultPath = "/peridynamics_results\0";  // Null-terminated string literal
        char[QueuePrefix.length + MaxQueueDigits + 1][] _controlQueuePaths;
        
        // Initialize queue paths with fixed-size buffers
        void initializeQueuePaths() {
            _controlQueuePaths = new char[QueuePrefix.length + MaxQueueDigits + 1][_numReplicas];
            foreach (i; 0.._numReplicas) {
                // Format directly into fixed buffer
                auto path = _controlQueuePaths[i][];
                auto len = snprintf(path.ptr, path.length, "%s%zu\0", QueuePrefix.ptr, i);
                assert(len > 0 && len < path.length, "Queue path formatting failed");
            }
        }
        
        mq_attr getQueueAttributes(mqd_t queue) {
            mq_attr attr;
            if (mq_getattr(queue, &attr) == -1) {
                throw new Exception(
                    format("Failed to get queue attributes: %s", 
                    fromStringz(strerror(errno)))
                );
            }
            return attr;
        }

        void verifyMessageSize(size_t messageSize, mqd_t queue, string messageType) {
            auto attr = getQueueAttributes(queue);
            if (messageSize > attr.mq_msgsize) {
                throw new Exception(
                    format("%s message size (%d bytes) exceeds queue capacity (%d bytes)",
                        messageType, messageSize, attr.mq_msgsize)
                );
            }
        }

        void calculateQueueAttributes(size_t numPositions) {
            // Read system limits
            auto maxMsgSize = readText("/proc/sys/fs/mqueue/msgsize_max").strip.to!size_t;
            
            // Calculate required size for state messages (larger than control messages)
            _msgSize = size_t.sizeof * 2 +  // replicaId + array length
                      double.sizeof * 2 +   // energy + multiplier
                      V.sizeof * numPositions;  // positions array

            if (_msgSize > maxMsgSize) {
                throw new Exception(format(
                    "Required message size (%d bytes) exceeds system limit (%d bytes). " ~
                    "Increase msg_size_max in /proc/sys/fs/mqueue/msgsize_max",
                    _msgSize, maxMsgSize
                ));
            }
        }

    public:
        this(size_t numPositions, size_t numReplicas) {
            calculateQueueAttributes(numPositions);

            // Configure queue attributes
            mq_attr attr;
            attr.mq_flags = 0;
            attr.mq_msgsize = _msgSize;
            attr.mq_maxmsg = totalCPUs * 2;  // Queue depth based on number of CPUs
            attr.mq_curmsgs = 0;

            // Initialize management fields
            _numReplicas = numReplicas;
            _controlQueues = new mqd_t[_numReplicas];
            
            // Initialize C string paths
            initializeQueuePaths();
            
            // Create control queues (one per worker)
            foreach (i; 0.._numReplicas) {
                _controlQueues[i] = mq_open(
                    _controlQueuePaths[i].ptr,
                    O_CREAT | O_RDWR,
                    S_IRUSR | S_IWUSR,
                    &attr
                );
                
                if (_controlQueues[i] == cast(mqd_t)-1) {
                    throw new Exception(format("Failed to create control queue for worker %d: %s",
                        i, fromStringz(strerror(errno))));
                }
            }
            
            // Create shared result queue
            _resultQueue = mq_open(
                ResultPath.ptr,
                O_CREAT | O_RDWR,
                S_IRUSR | S_IWUSR,
                &attr
            );
            
            if (_resultQueue == cast(mqd_t)-1) {
                // Calculate control message size
                ControlMessage!V tempCtrl;
                ubyte[] tempBuffer;
                tempCtrl.serialize(tempBuffer);
                size_t controlMsgSize = tempBuffer.length;

                string errorDetails = format(
                    "Failed to create message queues.\n" ~
                    "Error: %s\n" ~
                    "Requested configuration:\n" ~
                    "  Control message size: %d bytes\n" ~
                    "  State message size: %d bytes\n" ~
                    "  Queue depth: %d messages\n" ~
                    "Current system limits:\n" ~
                    "  msg_max: %s\n" ~
                    "  msgsize_max: %s\n" ~
                    "  queues_max: %s\n" ~
                    "\nPossible solutions:\n" ~
                    "1. Clean up stale queues: rm -f /dev/mqueue/peridynamics_*\n" ~
                    "2. Check queue directory permissions: ls -l /dev/mqueue\n" ~
                    "3. Increase system limits (as root):\n" ~
                    "   echo 8192 > /proc/sys/fs/mqueue/msgsize_max\n" ~
                    "   echo 10 > /proc/sys/fs/mqueue/msg_max\n" ~
                    "   echo 256 > /proc/sys/fs/mqueue/queues_max",
                    fromStringz(strerror(errno)),
                    controlMsgSize,
                    _msgSize,
                    totalCPUs * 2,
                    readText("/proc/sys/fs/mqueue/msg_max").strip,
                    readText("/proc/sys/fs/mqueue/msgsize_max").strip,
                    readText("/proc/sys/fs/mqueue/queues_max").strip
                );
                
                throw new Exception(errorDetails);
            }

            _initialized = true;
        }

        ~this() {
            if (_initialized) {
                // Close and unlink all control queues
                foreach (i; 0.._numReplicas) {
                    if (_controlQueues[i] != cast(mqd_t)-1) {
                        mq_close(_controlQueues[i]);
                        // Use stored C string path - guaranteed valid
                        mq_unlink(_controlQueuePaths[i].ptr);
                    }
                }
                
                // Close and unlink result queue
                if (_resultQueue != cast(mqd_t)-1) {
                    mq_close(_resultQueue);
                    mq_unlink(ResultPath.ptr);
                }
            }
        }

        // Master method - send control message to specific worker
        void sendControl(ControlMessage!V msg) {
            ubyte[] buffer;
            msg.serialize(buffer);
            
            // Use worker's dedicated control queue
            size_t workerId = msg.replicaId;
            verifyMessageSize(buffer.length, _controlQueues[workerId], "Control");
            
            if (mq_send(_controlQueues[workerId], cast(char*)buffer.ptr, buffer.length, 0) == -1) {
                string error = format("Failed to send control message to worker %d: %s",
                    workerId, fromStringz(strerror(errno)));
                throw new Exception(error);
            }
        }

        StateMessage!V receiveResult() {
            auto attr = getQueueAttributes(_resultQueue);
            ubyte[] buffer = new ubyte[attr.mq_msgsize];
            ssize_t size = mq_receive(
                _resultQueue, 
                cast(char*)buffer.ptr, 
                attr.mq_msgsize, 
                null
            );
            
            if (size == -1) {
                string error = format("Failed to receive result: %s", fromStringz(strerror(errno)));
                throw new Exception(error);
            }
                
            return StateMessage!V.deserialize(buffer[0..size]);
        }

        // Worker method - receive control message from worker's queue
        ControlMessage!V receiveControl(size_t workerId) {
            auto attr = getQueueAttributes(_controlQueues[workerId]);
            ubyte[] buffer = new ubyte[attr.mq_msgsize];
            ssize_t size = mq_receive(
                _controlQueues[workerId],
                cast(char*)buffer.ptr,
                attr.mq_msgsize,
                null
            );

            if (size == -1) {
                string error = format("Failed to receive control message for worker %d: %s",
                    workerId, fromStringz(strerror(errno)));
                throw new Exception(error);
            }

            auto msg = ControlMessage!V.deserialize(buffer[0..size]);
            if (msg.replicaId != workerId) {
                throw new Exception(format("Worker %d received message intended for worker %d",
                    workerId, msg.replicaId));
            }
            return msg;
        }

        void sendResult(StateMessage!V msg) {
            ubyte[] buffer;
            msg.serialize(buffer);
            
            verifyMessageSize(buffer.length, _resultQueue, "State");
            
            if (mq_send(_resultQueue, cast(char*)buffer.ptr, buffer.length, 0) == -1) {
                string error = format("Failed to send result: %s", fromStringz(strerror(errno)));
                throw new Exception(error);
            }
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
            0,  // Auto-detect number of processes based on CPU count
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

// Removed System V semaphore helper functions - using POSIX sem_wait/sem_post directly


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
 *     "min_temperature": 0.1,
 *     "max_temperature": 2.0
 *   }
 * }
 */
class ParallelTemperingSolver(T, V) : OptimizationSolver!(T, V) {
    private:
        size_t _numReplicas;
        size_t _numProcesses;
        double _minTemperature;
        double _maxTemperature;
        double _gradientStepSize = 1e-6;
        bool _queuesInitialized;
        
        // Resource management
        ProcessManager _processManager;
        MessageQueueManager!V _queueManager;

        // Initialize message queues if not already done
        void initializeQueues(size_t numPositions) {
            if (!_queuesInitialized) {
                _queueManager = new MessageQueueManager!V(numPositions, _numReplicas);
                _queuesInitialized = true;
            }
        }
        
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
        double[] reorderTemperatures(StateMessage!V[] states, double[] temperatures) {
            import std.algorithm : sort, map;
            
            // Collect all states and their energies
            struct StateEnergy {
                size_t stateIndex;
                double energy;
            }
            
            auto allStates = new StateEnergy[_numReplicas];
            foreach (i; 0.._numReplicas) {
                allStates[i] = StateEnergy(i, states[i].energy);
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
        
        // Run replica process
        void runReplicaProcess(size_t replicaId, V[] positions, double multiplier,
                             ObjectiveFunction!(T, V) objective) {
            // Initialize state
            auto currentPositions = positions.dup;
            auto currentEnergy = objective.evaluate(positions, multiplier);
            
            // Process main loop
            while (true) {
                // Wait for temperature assignment from dedicated queue
                auto control = _queueManager.receiveControl(replicaId);
                double temperature = control.temperature;
                
                // Generate proposal using random walk
                foreach (ref pos; positions) {
                    auto perturbation = V.zero();
                    foreach (dim; 0..V.dimension) {
                        auto unitVec = V.zero();
                        unitVec[dim] = 1.0;
                        unitVec = unitVec * (normal(0, sqrt(temperature)) * _gradientStepSize);
                        perturbation = perturbation + unitVec;
                    }
                    pos = pos + perturbation;
                }
                
                multiplier += normal(0, sqrt(temperature)) * _gradientStepSize;
                
                // Evaluate proposal
                double proposedEnergy = objective.evaluate(positions, multiplier);
                
                // Metropolis acceptance
                if (proposedEnergy > currentEnergy && 
                    uniform01() >= exp((currentEnergy - proposedEnergy) / temperature)) {
                    // Reject proposal
                    foreach (i; 0..positions.length) {
                        positions[i] = currentPositions[i];
                    }
                } else {
                    // Accept proposal
                    currentEnergy = proposedEnergy;
                    currentPositions[] = positions[];
                }
                
                // Send state update
                auto state = StateMessage!V(
                    replicaId,
                    currentEnergy,
                    multiplier,
                    currentPositions
                );
                _queueManager.sendResult(state);
            }
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
            _queuesInitialized = false;  // Start with queues uninitialized
        }
        
        ~this() {
            // Resources clean themselves up through their destructors
        }

        override OptimizationResult!(T, V) minimize(
            V[] initialPositions,
            double initialMultiplier,
            ObjectiveFunction!(T, V) objective
        ) {
            // Lazy initialization of message queues
            initializeQueues(initialPositions.length);
            
            // Initialize temperatures
            auto temperatures = initializeTemperatures();
            
            // Spawn worker processes
            _processManager = new ProcessManager(_numReplicas);
            _processManager.spawnWorkers((size_t i) {
                try {
                    writeln("Replica ", i, " (PID ", thisProcessID(), ") starting");
                    stdout.flush();
                    
                    runReplicaProcess(i, initialPositions.dup, initialMultiplier, objective);
                } catch (Exception e) {
                    stderr.writeln("Replica ", i, " error: ", e.msg);
                }
            });
            
            // Track best solution and collect states
            double bestEnergy = double.infinity;
            V[] bestPositions;
            double bestMultiplier;
            auto states = new StateMessage!V[_numReplicas];
            
            // Main optimization loop
            for (size_t iter = 0; iter < _maxIterations; ++iter) {
                // Send temperature assignments to all replicas
                foreach (i; 0.._numReplicas) {
                    auto msg = ControlMessage!V(
                        ControlMessageType.TEMPERATURE_ASSIGNMENT,
                        temperatures[i],
                        i
                    );
                    _queueManager.sendControl(msg);
                }
                
                // Collect results from all replicas
                foreach (i; 0.._numReplicas) {
                    states[i] = _queueManager.receiveResult();
                    
                    // Update best solution if needed
                    if (states[i].energy < bestEnergy) {
                        bestEnergy = states[i].energy;
                        bestMultiplier = states[i].multiplier;
                        bestPositions = states[i].positions.dup;
                    }
                }
                
                // Check convergence
                double minEnergy = double.infinity;
                foreach (state; states) {
                    if (state.energy < minEnergy) {
                        minEnergy = state.energy;
                    }
                }
                
                if (iter > 0 && abs(minEnergy - bestEnergy) < _tolerance) {
                    break;
                }
                
                // Reorder temperatures based on energies
                temperatures = reorderTemperatures(states, temperatures);
            }
            
            // All results have been collected, so we can terminate the replicas
            writeln("All replica results collected, terminating replicas...");
            stdout.flush();
            _processManager.cleanupProcesses();
            
            return OptimizationResult!(T, V)(bestPositions, bestMultiplier);
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

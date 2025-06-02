module core.simulation;

import core.material_point;
import core.material_body;
import core.integration;
import math.vector;
import std.algorithm : min, max;
import std.math : sqrt, abs, exp, log, tanh, isNaN;
import std.exception : enforce;
import std.format : format;
import io.simulation_loader : OutputConfig;

// Interface for time step calculation strategies
interface TimeStepStrategy(T, V) {
    double calculateTimeStep(MaterialBody!(T, V) body);
}

// Adaptive time step implementation
class AdaptiveTimeStep(T, V) : TimeStepStrategy!(T, V) {
    private double maxTimeStep;          // [time]
    private double absoluteMaxTimeStep;  // [time]
    private double horizon;              // [length]
    private double safetyFactor;         // [] (dimensionless)
    private double maxRelativeMotion;    // [] (dimensionless ratio)
    private double characteristicVelocity; // [length/time]
    private double responseTimeScaling;    // [] (dimensionless ratio for response time)
    private double timeStepHistory;       // [time] Current time step
    private bool firstStep = true;        // Flag for first step
    
    this(
        double initialTimeStep,           // [time]
        double absoluteMaxTimeStep,       // [time]
        double horizon,                   // [length]
        double characteristicVelocity,    // [length/time]
        double responseTimeScaling,       // [] (ratio of response time to time step)
        double safetyFactor = 0.1,       // []
        double maxRelativeMotion = 0.01   // []
    ) {
        maxTimeStep = initialTimeStep;
        this.absoluteMaxTimeStep = absoluteMaxTimeStep;
        assert(maxTimeStep > 0.0, "Initial time step must be greater than zero");
        assert(absoluteMaxTimeStep > 0.0, "Absolute maximum time step must be greater than zero");
        this.horizon = horizon;
        this.characteristicVelocity = characteristicVelocity;
        this.responseTimeScaling = responseTimeScaling;
        assert(responseTimeScaling > 1.0, 
            "Response scaling must be greater than 1.0 to ensure response time exceeds time step");
        this.safetyFactor = safetyFactor;
        this.maxRelativeMotion = maxRelativeMotion;
        timeStepHistory = initialTimeStep;
    }
    
    // Smooth approximation functions
    private double smoothMax(double a, double b, double responseTime) {
        double scaled_a = a / responseTime;
        double scaled_b = b / responseTime;
        double max_val = max(scaled_a, scaled_b);
        double softMax = responseTime * (max_val + log(exp(scaled_a - max_val) + exp(scaled_b - max_val)));
        if (softMax > max(a, b) || isNaN(softMax)) {
            return max(a, b);
        }
        return softMax;
    }

    private double smoothMin(double a, double b, double responseTime) {
        return -smoothMax(-a, -b, responseTime);
    }
    
    private double lerp(double a, double b, double t) {
        return a * (1 - t) + b * t;
    }

    double calculateTimeStep(MaterialBody!(T, V) body) {
        if (firstStep) {
            firstStep = false;
            return timeStepHistory;  // Return initial time step
        }
        
        if (body.numPoints == 0) return maxTimeStep;
        
        // Calculate response time based on current time step
        double responseTime = responseTimeScaling * timeStepHistory;
        
        // Find maximum velocity magnitude and minimum spacing
        double maxVelocity = 0.0;
        double minSpacing = double.max;
        
        T[] neighbors;
        for (size_t i = 0; i < body.numPoints; ++i) {
            auto point = body[i];
            maxVelocity = smoothMax(maxVelocity, point.velocity.magnitude, responseTime);
            
            body.neighbors(i, neighbors);
            foreach (neighbor; neighbors) {
                double spacing = (point.position - neighbor.position).magnitude;
                minSpacing = smoothMin(minSpacing, spacing, responseTime);
            }
        }
        
        // Calculate time step constraints using smooth operations
        double dtCFL = safetyFactor * (minSpacing / maxVelocity);
        double dtDisp = safetyFactor * (horizon * maxRelativeMotion / maxVelocity);
        
        // Apply constraints with smooth transitions
        double baseStep = smoothMin(dtCFL, dtDisp, responseTime);
        double targetStep = smoothMin(baseStep, absoluteMaxTimeStep, responseTime);
        
        // Calculate stability metric
        double stabilityMetric = 1.0 / (1.0 + (maxVelocity/characteristicVelocity)^^2);
        
        // Update time step with smooth transition
        double newStep = lerp(targetStep, absoluteMaxTimeStep, stabilityMetric);
        timeStepHistory = timeStepHistory + (newStep - timeStepHistory) / responseTimeScaling;
        
        enforce(timeStepHistory > 0.0, "Time step must be greater than zero");
        return timeStepHistory;
    }
}

// Generic simulation function with adaptive time stepping and Lagrangian integration
void simulate(T, V)(
    MaterialBody!(T, V) body,
    TimeStepStrategy!(T, V) timeStepStrategy,
    LagrangianIntegrator!(T, V) integrator,
    double totalTime,
    OutputConfig output
) if (isMaterialPoint!(T, V)) {
    import std.stdio : writefln;
    import std.path : stripExtension, baseName, dirName, buildPath;
    import std.file : copy;
    
    size_t step = 0;
    double currentTime = 0.0;
    double lastTimeStep = 0.0;
    double nextOutputTime = 0.0;
    string lastOutputFile;
    
    // Get base filename and directory
    string outputDir = dirName(output.csv_file);
    string baseFile = stripExtension(baseName(output.csv_file));
    
    // Ensure intermediate files go to the same directory as final output
    string getOutputPath(string filename) {
        return buildPath(outputDir, filename);
    }
    
    // Main time stepping loop
    while (currentTime < totalTime) {
        // Calculate adaptive time step
        double timeStep = timeStepStrategy.calculateTimeStep(body);
        
        // Log time step changes if they differ significantly
        if (abs(timeStep - lastTimeStep) / lastTimeStep > 0.0) {
            writefln("Time %.4e: Adjusted step size to %.4e", currentTime, timeStep);
        }
        lastTimeStep = timeStep;
        
        // Ensure we don't exceed totalTime
        if (currentTime + timeStep > totalTime) {
            timeStep = totalTime - currentTime;
        }
        
        // Handle output based on configuration
        bool shouldOutput = false;
        string currentOutputFile;
        
        if (output.step_interval > 0 && step % output.step_interval == 0) {
            // Step-based output
            currentOutputFile = getOutputPath(format("%s_step%05d.csv", baseFile, step));
            shouldOutput = true;
        }
        else if (output.time_interval > 0 && currentTime >= nextOutputTime) {
            // Time-based output
            currentOutputFile = getOutputPath(format("%s_t%09d.csv", baseFile, 
                cast(size_t)(currentTime * 1e9)));
            shouldOutput = true;
            nextOutputTime += output.time_interval;
        }
        
        if (shouldOutput) {
            body.exportToCSV(currentOutputFile);
            lastOutputFile = currentOutputFile;
        }
        
        // Update system state using integrator
        integrator.integrate(body, timeStep);
        
        currentTime += timeStep;
        ++step;
    }
    
    // Always save final state to requested output file
    if (lastOutputFile) {
        copy(lastOutputFile, output.csv_file);
    }
}

// Unit tests
// Base test point implementation
class TestPoint(V) : MaterialPoint!(TestPoint!V, V) {
    protected V _pos;
    protected V _refPos;
    protected V _vel;
    
    this(V pos, V vel = V.zero()) {
        _pos = pos;
        _refPos = pos;
        _vel = vel;
    }
    
    @property V position() const { return _pos; }
    @property V referencePosition() const { return _refPos; }
    @property V velocity() const { return _vel; }
    
    void updateState(const(TestPoint!V)[] neighbors, double timeStep) {
        _pos = _pos + _vel * timeStep;  // Update position using current velocity
    }
}

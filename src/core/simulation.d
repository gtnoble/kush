module core.simulation;

import core.material_point;
import core.material_body;
import math.vector;
import std.algorithm : min, max;
import std.math : sqrt, abs;

// Interface for time step calculation strategies
interface TimeStepStrategy(T, V) {
    double calculateTimeStep(const MaterialBody!(T, V) body);
}

// Adaptive time step implementation
class AdaptiveTimeStep(T, V) : TimeStepStrategy!(T, V) {
    private double maxTimeStep;
    private double safetyFactor;
    private double horizon;
    private double maxAllowedRelativeMotion;
    private double initialTimeStep;
    private bool firstStep = true;
    
    this(double maxDt, double horizon, double safety = 0.1, double maxRelMotion = 0.1, double initialDt = -1.0) {
        maxTimeStep = maxDt;
        safetyFactor = safety;
        this.horizon = horizon;
        maxAllowedRelativeMotion = maxRelMotion;
        initialTimeStep = initialDt;
    }
    
    double calculateTimeStep(const MaterialBody!(T, V) body) {
        if (body.numPoints == 0) return maxTimeStep;
        
        // Use initial time step for first iteration if specified
        if (firstStep && initialTimeStep >= 0.0) {
            firstStep = false;
            return min(initialTimeStep, maxTimeStep);
        }
        firstStep = false;
        
        // Find maximum velocity magnitude
        double maxVelocity = 0.0;
        double minSpacing = double.max;
        
        // Calculate maximum velocity and minimum spacing
        for (size_t i = 0; i < body.numPoints; ++i) {
            auto point = body[i];
            double velMag = point.velocity.magnitude;
            maxVelocity = max(maxVelocity, velMag);
            
            // Find minimum spacing between this point and its neighbors
            const(T)[] neighbors = body.neighbors(i);
            foreach (neighbor; neighbors) {
                double spacing = (point.position - neighbor.position).magnitude;
                minSpacing = min(minSpacing, spacing);
            }
        }
        
        if (maxVelocity < 1e-10) return maxTimeStep;  // If essentially static
        
        // Calculate time steps based on both criteria
        double dtCFL = safetyFactor * (minSpacing / maxVelocity);
        double dtDisp = safetyFactor * (horizon * maxAllowedRelativeMotion / maxVelocity);
        
        // Return minimum of all constraints
        return min(min(dtCFL, dtDisp), maxTimeStep);
    }
}

// Generic simulation function with adaptive time stepping
void simulate(T, V)(
    MaterialBody!(T, V) body,
    TimeStepStrategy!(T, V) timeStepStrategy,
    double totalTime
) if (isMaterialPoint!(T, V)) {
    import std.stdio : writefln;
    size_t step = 0;
    double currentTime = 0.0;
    double lastTimeStep = 0.0;
    
    // Main time stepping loop
    while (currentTime < totalTime) {
        // Calculate adaptive time step
        double timeStep = timeStepStrategy.calculateTimeStep(body);
        
        // Log time step changes if they differ significantly
        if (abs(timeStep - lastTimeStep) > 1e-10) {
            writefln("Time %.4e: Adjusted step size to %.4e", currentTime, timeStep);
        }
        lastTimeStep = timeStep;
        
        // Ensure we don't exceed totalTime
        if (currentTime + timeStep > totalTime) {
            timeStep = totalTime - currentTime;
        }
        
        // Export state at every step
        import std.format : format;
        string filename = format("simulation_step_%04d.csv", step);
        body.exportToCSV(filename);

        // Update each point
        for (size_t i = 0; i < body.numPoints; ++i) {
            auto point = body[i];  // Uses non-const opIndex
            const(T)[] neighbors = body.neighbors(i);
            point.updateState(neighbors, timeStep);
        }
        
        currentTime += timeStep;
        ++step;
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

// Test constant velocity motion
unittest {
    import std.math : abs;
    
    // Test point moving with constant velocity
    auto p1 = new TestPoint!Vector1D(Vector1D(0.0), Vector1D(1.0));  // Initial position 0, velocity 1
    auto body = new MaterialBody!(TestPoint!Vector1D, Vector1D)([p1], 1.0);
    
    // Create adaptive time step strategy
    auto timeStepStrategy = new AdaptiveTimeStep!(TestPoint!Vector1D, Vector1D)(0.2, 1.0, 0.1);
    
    // Simulate for 1.0 total time with adaptive stepping
    simulate(body, timeStepStrategy, 1.0);
    
    // After 1 second of simulation, position should be 1.0
    assert(abs(body[0].position[0] - 1.0) < 1e-10);
}

// Test variable velocity motion
unittest {
    import std.math : abs, PI;
    
    // Oscillating point implementation
    class OscillatingPoint(V) : TestPoint!V {
        private double frequency;
        private double currentTime = 0.0;
        
        this(V pos, double freq) {
            super(pos);
            frequency = freq;
        }
        
        override void updateState(const(TestPoint!V)[] neighbors, double timeStep) {
            // v(t) = A*cos(ωt), position is integral of velocity
            double A = 1.0;  // Amplitude
            currentTime += timeStep;
            
            _vel = V([A * cos(frequency * currentTime)]);
            _pos = V([A/frequency * sin(frequency * currentTime)]);
        }
    }
    
    // Create oscillating point with 1 Hz frequency
    auto oscillator = new OscillatingPoint!Vector1D(Vector1D(0.0), 2.0 * PI);
    auto body = new MaterialBody!(TestPoint!Vector1D, Vector1D)([oscillator], 1.0);
    
    // Create adaptive stepper with smaller max time step
    auto timeStepStrategy = new AdaptiveTimeStep!(TestPoint!Vector1D, Vector1D)(0.05, 1.0, 0.1);
    
    // Simulate for one period
    simulate(body, timeStepStrategy, 1.0);
    
    // Position should be close to 0 after one period
    assert(abs(oscillator.position[0]) < 1e-3);
}

// Test initial time step specification
unittest {
    import std.math : abs;
    
    // Create a test point with constant velocity
    auto p1 = new TestPoint!Vector1D(Vector1D(0.0), Vector1D(1.0));
    auto body = new MaterialBody!(TestPoint!Vector1D, Vector1D)([p1], 1.0);
    
    // Create adaptive stepper with initial time step
    double initialDt = 0.1;
    auto timeStepStrategy = new AdaptiveTimeStep!(TestPoint!Vector1D, Vector1D)(0.2, 1.0, 0.1, 0.1, initialDt);
    
    // First step should use initial time step
    double firstStep = timeStepStrategy.calculateTimeStep(body);
    assert(abs(firstStep - initialDt) < 1e-10, "First time step should match initial time step");
    
    // Second step should use adaptive calculation
    double secondStep = timeStepStrategy.calculateTimeStep(body);
    assert(abs(secondStep - initialDt) > 1e-10, "Second time step should differ from initial time step");
}

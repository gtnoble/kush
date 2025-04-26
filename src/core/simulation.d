module core.simulation;

import core.material_point;
import core.material_body;
import math.vector;

// Generic simulation function
void simulate(T, V)(MaterialBody!(T, V) body, double timeStep, size_t numSteps)
    if (isMaterialPoint!(T, V)) {
    // Main time stepping loop
    for (size_t step = 0; step < numSteps; ++step) {
        // Update each point directly
        for (size_t i = 0; i < body.numPoints; ++i) {
            T point = body[i];
            T[] neighbors = body.neighbors(i);
            point.updateState(neighbors, timeStep);
        }
    }
}

// Unit tests
unittest {
    import std.math : abs;
    
    // Create a simple test material point implementation
    class TestPoint(V) : MaterialPoint!(TestPoint!V, V) {
        private V _pos;
        private V _refPos;
        private V _vel;
        
        this(V pos, V vel = V.zero()) {
            _pos = pos;
            _refPos = pos;
            _vel = vel;
        }
        
        @property V position() const { return _pos; }
        @property V referencePosition() const { return _refPos; }
        
        void updateState(TestPoint!V[] neighbors, double timeStep) {
            _pos = _pos + _vel * timeStep;  // Update position using current velocity
        }
    }
    
    // Test with 1D constant velocity motion
    auto p1 = new TestPoint!Vector1D(Vector1D(0.0), Vector1D(1.0));  // Initial position 0, velocity 1
    auto points = [p1];
    auto body = new MaterialBody!(TestPoint!Vector1D, Vector1D)(points, 1.0);
    
    // Simulate for 10 steps with dt = 0.1
    simulate(body, 0.1, 10);
    
    // After 1 second (10 steps * 0.1), position should be 1.0
    assert(abs(body[0].position[0] - 1.0) < 1e-10);
}

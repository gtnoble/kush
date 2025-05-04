module core.velocity_constraint;

import math.vector;
import core.material_body;
import core.material_point;
import std.math : sqrt;

// Relativistic constraint to prevent velocities exceeding speed of light
struct RelativisticConstraint(V) if (is(V == Vector!N, size_t N)) {
    enum C = 299_792_458.0;  // Speed of light in m/s
    //enum C = 6320.0;  // Speed of sound in aluminum in m/s
    
    static double evaluateConstraint(V velocity) {
        double v2 = velocity.magnitudeSquared();
        return v2 > C * C ? (sqrt(v2) - C) * (sqrt(v2) - C) : 0.0;
    }
}

// Velocity constraint for a single point
struct VelocityConstraint(V) if (is(V == Vector!N, size_t N)) {
    private:
        V _targetVelocity;   // Target velocity vector

    public:
        this(V target) {
            _targetVelocity = target;
        }
        
        @property ref V targetVelocity() { return _targetVelocity; }

        // Evaluate squared magnitude of velocity difference
        double evaluateConstraint(V velocity) const {
            V diff = velocity - _targetVelocity;
            return diff.magnitudeSquared();  // |v - v_target|²
        }
}

// System-wide velocity constraint functions
struct SystemVelocityConstraint(T, V) if (isMaterialPoint!(T, V)) {
    // Evaluate total constraint violation for all constrained points
    static double evaluateSystemConstraint(MaterialBody!(T, V) body, V[] proposedVelocities = null) {
        double totalViolation = 0.0;
        
        for (size_t i = 0; i < body.numPoints; ++i) {
            auto point = body[i];
            V velocity = proposedVelocities ? proposedVelocities[i] : point.velocity;

            // Add existing velocity constraint violations
            if (point.velocityConstraint !is null) {
                totalViolation += point.velocityConstraint.evaluateConstraint(velocity);
            }
            
            // Add relativistic constraint violation
            totalViolation += RelativisticConstraint!V.evaluateConstraint(velocity);
        }
        
        return totalViolation;
    }

    // Calculate total constraint energy contribution using single scalar multiplier
    static double systemEnergyContribution(MaterialBody!(T, V) body, double multiplier, V[] proposedVelocities = null) {
        return multiplier * evaluateSystemConstraint(body, proposedVelocities);
    }
}

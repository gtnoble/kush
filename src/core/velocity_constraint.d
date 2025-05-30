module core.velocity_constraint;

import math.vector;
import core.material_body;
import core.material_point;
import std.math : sqrt;

// Individual constraint term for a point's velocity component
struct ConstraintTerm(V) if (isVector!V) {
    size_t pointIndex;      // Which point this constraint applies to
    size_t componentIndex;  // Which component (x,y,z) this constraint applies to
    double targetComponent; // Target velocity for this component
    
    double evaluate(V velocity) const {
        return velocity[componentIndex] - targetComponent;  // Simple component difference
    }
}

// Velocity constraint for a single point
struct VelocityConstraint(V) if (isVector!V) {
    private:
        V _targetVelocity;   // Target velocity vector

    public:
        this(V target) {
            _targetVelocity = target;
        }
        
        @property V targetVelocity() const { return _targetVelocity; }

        // Evaluate squared magnitude of velocity difference
        double evaluateConstraint(V velocity) const {
            V diff = velocity - _targetVelocity;
            return diff.magnitudeSquared();  // |v - v_target|²
        }
}

// System-wide velocity constraint functions
struct SystemVelocityConstraint(T, V) if (isMaterialPoint!(T, V)) {
    // Get array of individual constraint terms - one per velocity component
    static ConstraintTerm!V[] getSystemConstraints(MaterialBody!(T, V) body) {
        ConstraintTerm!V[] terms;
        
        // Add component-wise constraints for points that have velocity targets
        for (size_t i = 0; i < body.numPoints; ++i) {
            if (body[i].velocityConstraint !is null) {
                auto targetVel = body[i].velocityConstraint.targetVelocity;
                // Create a separate constraint for each component
                foreach (j; 0..V.length) {
                    terms ~= ConstraintTerm!V(
                        i,           // point index
                        j,           // component index
                        targetVel[j] // target velocity component
                    );
                }
            }
        }
        
        return terms;
    }
}

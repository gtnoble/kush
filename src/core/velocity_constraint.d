module core.velocity_constraint;

import math.vector;
import core.material_body;
import core.material_point;
import std.math : sqrt;

// Individual constraint term for a point
struct ConstraintTerm(V) if (is(V == Vector!N, size_t N)) {
    size_t pointIndex;  // Which point this constraint applies to
    V targetVelocity;   // Target velocity for this point
    
    double evaluate(V velocity) const {
        V diff = velocity - targetVelocity;
        return diff.magnitudeSquared();  // |v - v_target|²
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
        
        @property V targetVelocity() const { return _targetVelocity; }

        // Evaluate squared magnitude of velocity difference
        double evaluateConstraint(V velocity) const {
            V diff = velocity - _targetVelocity;
            return diff.magnitudeSquared();  // |v - v_target|²
        }
}

// System-wide velocity constraint functions
struct SystemVelocityConstraint(T, V) if (isMaterialPoint!(T, V)) {
    // Get array of individual constraint terms
    static ConstraintTerm!V[] getSystemConstraints(MaterialBody!(T, V) body) {
        ConstraintTerm!V[] terms;
        
        // Only add velocity target constraints for points that have them
        for (size_t i = 0; i < body.numPoints; ++i) {
            if (body[i].velocityConstraint !is null) {
                terms ~= ConstraintTerm!V(
                    i,
                    body[i].velocityConstraint.targetVelocity
                );
            }
        }
        
        return terms;
    }
}

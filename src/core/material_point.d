module core.material_point;

import math.vector;
import core.velocity_constraint;

// Generic MaterialPoint interface that takes both the concrete type T and vector type V
interface MaterialPoint(T, V) if (isVector!V) {
    // Position properties
    @property V position() const;
    @property V referencePosition() const;
    @property V velocity() const;
    @property void velocity(V vel);

    // Velocity constraint
    @property const(VelocityConstraint!V*) velocityConstraint() const;
    @property void velocityConstraint(VelocityConstraint!V* constraint);

    // Mass property
    @property double mass() const;

    // Lagrangian calculation
    double computeLagrangian(const(T)[] neighbors, V proposedPosition, double timeStep) const;
    
    // State update methods
    void position(V newPosition);
}

// Helper template to verify a type implements MaterialPoint interface
enum isMaterialPoint(T, V) = is(T : MaterialPoint!(T, V)) && isVector!V;

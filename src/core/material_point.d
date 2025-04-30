module core.material_point;

import math.vector;

// Generic MaterialPoint interface that takes both the concrete type T and vector type V
interface MaterialPoint(T, V) if (is(V == Vector!N, size_t N)) {
    // Position properties
    @property V position() const;
    @property V referencePosition() const;
    @property V velocity() const;
    @property void velocity(V vel);

    // Lagrangian calculation
    double computeLagrangian(const(T)[] neighbors, V proposedPosition, double timeStep) const;
    
    // State update methods
    void updateState(const(T)[] neighbors, double timeStep);
    void position(V newPosition);
}

// Helper template to constrain vector dimensions
enum isVector(V) = is(V == Vector!N, size_t N);

// Helper template to verify a type implements MaterialPoint interface
enum isMaterialPoint(T, V) = is(T : MaterialPoint!(T, V)) && isVector!V;

// Unit tests
unittest {
    // Test compile-time constraints
    static assert(isVector!(Vector1D));
    static assert(isVector!(Vector2D));
    static assert(isVector!(Vector3D));
    static assert(!isVector!(int));
    
    // Note: Full implementation testing will be done with concrete types
}

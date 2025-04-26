module models.bond_based_point;

import core.material_point;
import math.vector;
import std.math : abs;

// Bond-based material point implementation
class BondBasedPoint(V) : MaterialPoint!(BondBasedPoint!V, V)
    if (isVector!V) {
    private V _position;
    private V _referencePosition;
    private V _velocity;

    // Velocity accessor
    @property V velocity() const {
        return _velocity;
    }

    // Velocity setter
    @property void velocity(V vel) {
        _velocity = vel;
    }
    private double _mass;
    
    // Material properties
    private double _bondStiffness;  // Bond stiffness constant
    private double _criticalStretch;  // Critical stretch for bond breaking
    
    // Constructor
    this(V refPos, double pointMass, double bondStiffness, double criticalStretch) {
        _referencePosition = refPos;
        _position = refPos;
        _velocity = V.zero();
        _mass = pointMass;
        _bondStiffness = bondStiffness;
        _criticalStretch = criticalStretch;
    }
    
    // Position properties
    @property V position() const {
        return _position;
    }
    
    @property V referencePosition() const {
        return _referencePosition;
    }
    
    // Calculate bond force between two points with reversible damage
    private V bondForce(const(BondBasedPoint!V) neighbor) const {
        // Calculate reference and current vectors between points
        V refVector = neighbor.referencePosition - _referencePosition;
        V curVector = neighbor.position - _position;
        
        double refLength = refVector.magnitude();
        double curLength = curVector.magnitude();
        
        // Calculate stretch
        double stretch = (curLength - refLength) / refLength;
        
        // Reversible damage model: zero force if stretch exceeds critical value
        if (abs(stretch) > _criticalStretch) {
            return V.zero();
        }
        
        // Bond force calculation (linear micropotential)
        return curVector.unit() * (_bondStiffness * stretch);
    }
    
    // Direct state update implementation
    void updateState(const(BondBasedPoint!V)[] neighbors, double timeStep) {
        // Calculate total force
        V totalForce = V.zero();
        foreach (neighbor; neighbors) {
            totalForce = totalForce + bondForce(neighbor);
        }
        
        // Update velocity using acceleration (F = ma)
        _velocity = _velocity + totalForce * (timeStep / _mass);
        
        // Update position using new velocity
        _position = _position + _velocity * timeStep;
    }
}

// Unit tests
unittest {
    import std.math : abs, PI;
    
    // Test 1D stretching
    {
        // Create two points with a bond
        double mass = 1.0;
        double bondStiffness = 1.0;
        double criticalStretch = 0.1;
        
        auto p1 = new BondBasedPoint!Vector1D(Vector1D(0.0), mass, bondStiffness, criticalStretch);
        auto p2 = new BondBasedPoint!Vector1D(Vector1D(1.0), mass, bondStiffness, criticalStretch);
        
        // Update state for p1 with p2 as neighbor
        p1.updateState([p2], 0.1);
        
        // Due to initial separation of 1.0, there should be no force and thus no position/velocity change
        assert(abs(p1.velocity[0]) < 1e-10);
        assert(abs(p1.position[0]) < 1e-10);
    }
    
    // Test 2D bond breaking
    {
        double mass = 1.0;
        double bondStiffness = 1000.0;
        double criticalStretch = 0.1;
        
        // Create two points initially at rest
        auto p1 = new BondBasedPoint!Vector2D(Vector2D(0.0, 0.0), mass, bondStiffness, criticalStretch);
        auto p2 = new BondBasedPoint!Vector2D(Vector2D(1.0, 0.0), mass, bondStiffness, criticalStretch);
        
        // Move second point to create stretch > critical
        p2._position = Vector2D(1.2, 0.0);  // 20% stretch
        
        // Update state for p1
        p1.updateState([p2], 0.1);
        
        // Force and resulting changes should be zero due to damage
        assert(abs(p1.velocity[0]) < 1e-10);
        assert(abs(p1.velocity[1]) < 1e-10);
    }
}

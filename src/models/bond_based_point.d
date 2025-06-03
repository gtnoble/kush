module models.bond_based_point;

import core.material_point;
import core.damper;
import core.velocity_constraint : VelocityConstraint;
import math.vector;
import std.math : abs, exp, sinh;
import std.complex : exp;
import std.typecons : Nullable;
import std.exception : enforce;

const double SPEED_OF_LIGHT = 299792458.0; // Speed of light in m/s
const double PLANCK_LENGTH = 1.616255e-35; // Planck length in meters

// Bond-based material point implementation
class BondBasedPoint(V) : MaterialPoint!(BondBasedPoint!V, V)
    if (isVector!V) {
    private V _position;
    private V _referencePosition;
    private V _velocity;
    private double _mass;
    private double _volume;
    private V _constantForce;  // Store constant external force
    private double _timeElapsed;      // Track simulation time
    private VelocityConstraint!V* _velocityConstraint;  // Velocity constraint (pointer to allow null checks)
    private double _rampDuration;     // Time to reach target force

    // Interface implementation
    @property V position() const { return _position; }
    
    @property const(VelocityConstraint!V*) velocityConstraint() const {
        return _velocityConstraint;
    }
    
    @property void velocityConstraint(VelocityConstraint!V* constraint) {
        _velocityConstraint = constraint;
    }
    @property V referencePosition() const { return _referencePosition; }
    @property V velocity() const { return _velocity; }
    @property void velocity(V vel) { _velocity = vel; }
    @property void position(V newPos) { _position = newPos; }
    @property double mass() const { return _mass; }
    
    // Material properties
    private double _bondStiffness;  // Bond stiffness constant
    private double _criticalStretch;  // Critical stretch for bond breaking
    private Damper!V _damper;      // Damping strategy

    // Constructor
    this(
        V refPos,
        double pointMass,
        double pointVolume,
        double bondStiffness,
        double criticalStretch,
        Damper!V damper,               // Damping strategy
        V initialVelocity = V.zero(),
        bool fixedVelocity = false,
        V constantForce = V.zero(),     // Target force at end of ramp (already scaled by point count)
        double rampDuration = 1e-6    // Duration of force ramp (default 1μs)
    ) {
        _referencePosition = refPos;
        _position = refPos;
        _velocity = initialVelocity;
        _mass = pointMass;
        _volume = pointVolume;
        _bondStiffness = bondStiffness;
        _criticalStretch = criticalStretch;
        _damper = damper;
        if (fixedVelocity) {
            _velocityConstraint = new VelocityConstraint!V(initialVelocity);
        } else {
            _velocityConstraint = null;  // No constraint
        }
        _rampDuration = rampDuration;
        _timeElapsed = 0.0;
        _constantForce = constantForce;  // Start with zero force
    }

// Generic Lagrangian computation that works with both real and complex numbers
T computeLagrangianGeneric(T)(const(BondBasedPoint!V)[] neighbors, Vector!(T, V.length) proposedPosition, double timeStep) const {
    // Convert real position to match type of proposedPosition
    auto pos = _position;
    auto refPos = _referencePosition;
    
    // Kinetic energy
    auto proposedVelocity = (proposedPosition - pos) / T(timeStep);
    T kineticEnergy = T(0.5) * T(_mass) * proposedVelocity.magnitudeSquared();
    
    // Potential energy from bonds
    T potentialEnergy = T(0);
    foreach (neighbor; neighbors) {
        // Convert neighbor positions to match type of proposedPosition
        auto refVector = neighbor.referencePosition - refPos;
        auto propVector = neighbor.position - proposedPosition;
        
        auto refLength = refVector.magnitude();
        auto propLength = propVector.magnitude();
        
        // Calculate stretch
        auto stretch = (propLength - refLength) / refLength;
        auto c = T(_criticalStretch);
        auto k = T(_bondStiffness);
        
        // Calculate both energy density terms
        auto V_quad = T(0.5) * k * stretch * stretch * _volume;
        auto V_const = T(0.5) * k * c * c * _volume;
        
        // Smooth transition factor (0 to 1)
        //auto alpha = T(1) / (T(1) + exp(T(20) * (stretch - c) / c));
        auto alpha = T(1);
        
        // Blend energies smoothly. Half the bond energy is used to avoid double counting
        potentialEnergy += _volume * (alpha * V_quad + (T(1) - alpha) * V_const) * 0.5;
    }
    
    // Dissipation calculation
    T totalDissipation = T(0);
    totalDissipation += _damper.calculateGlobalDissipation!(T, typeof(proposedVelocity))(proposedVelocity, _mass, timeStep);
    
    foreach (neighbor; neighbors) {
        // Convert positions to match type of proposedPosition for damping
        auto propVector = neighbor.position - proposedPosition;
        auto refVector = neighbor.referencePosition - refPos;
        auto displacement = propVector - refVector;
        auto relativeVelocity = neighbor.velocity - proposedVelocity;
        
        totalDissipation += _damper.calculateBondDissipation!(T, typeof(relativeVelocity))(
            relativeVelocity,
            displacement,
            _mass,
            timeStep
        ) * 0.5; // Half the dissipation to avoid double counting
    }
    
    // External force contribution
    potentialEnergy += _constantForce.dot(proposedPosition - refPos);
    
    return kineticEnergy - potentialEnergy - totalDissipation;
}

// Modified to use generic implementation
    double computeLagrangian(const(BondBasedPoint!V)[] neighbors, V proposedPosition, double timeStep) const {
        return computeLagrangianGeneric!double(neighbors, proposedPosition, timeStep);
}

// Implement gradient using complex step differentiation
    V computeLagrangianGradient(const(BondBasedPoint!V)[] neighbors, V proposedPosition, double timeStep) const {
        import std.complex : Complex; 
        alias C = Complex!double;
    
    const double h = 1e-200;  // Complex step size
    //const double h = 1e-19;  // Complex step size
    V result = V.zero();
    
    // Compute each component of the gradient using complex-step differentiation
    for (size_t i = 0; i < V.length; i++) {
        // Create perturbed position with imaginary step
        auto perturbedPos = Vector!(C, V.length)();
        foreach (j; 0..V.length) {
            perturbedPos[j] = C(proposedPosition[j], j == i ? h : 0);
        }
        
        // Compute Lagrangian with complex step
        auto lagrangian = computeLagrangianGeneric!C(neighbors, perturbedPos, timeStep);
        
        // Extract gradient component from imaginary part
        result[i] = lagrangian.im / h;
    }
    
    return result;
}
    
    // Check if a bond is under compression using displacement direction
    private bool isCompressing(const(BondBasedPoint!V) neighbor) const {
        V relativeVelocity = neighbor.velocity - _velocity;
        V refVector = neighbor.referencePosition - _referencePosition;
        V curVector = neighbor.position - _position;
        V displacement = curVector - refVector;
        return relativeVelocity.dot(displacement) < 0;
    }
}

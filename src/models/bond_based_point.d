module models.bond_based_point;

import core.material_point;
import math.vector;
import std.math : abs;
import std.typecons : Nullable;
import std.exception : enforce;

const double SPEED_OF_LIGHT = 299792458.0; // Speed of light in m/s
const double PLANCK_LENGTH = 1.616255e-35; // Planck length in meters

// Unified damper interface that handles both global and bond damping
interface Damper(V) if (isVector!V) {
    // Calculate global damping force
    V calculateGlobalForce(V velocity, double mass) const;
    
    // Calculate per-bond damping force
    V calculateBondForce(
        V relativeVelocity,
        V bondVector,
        double mass
    ) const;

    // Calculate global damping dissipation
    double calculateGlobalDissipation(V velocity, double mass, double timeStep) const;
    
    // Calculate per-bond damping dissipation
    double calculateBondDissipation(
        V relativeVelocity,  // Relative velocity between points
        V bondVector,        // Vector from point to neighbor
        double mass,
        double timeStep      // Time step for converting power to energy
    ) const;
}

alias InfluenceFunction = double function(double bondLength, double neighborhoodRadius);

// Cubic spline influence function for bond-based models
double cubicSplineInfluence(double bondLength, double neighborhoodRadius) {
    double q = bondLength / neighborhoodRadius;
    return 1.0 - 3.0 * q * q + 2.0 * q * q * q;
} 

// Current implementation combining exponential damping and Monaghan viscosity
class StandardDamper(V) : Damper!V if (isVector!V) {
    private double _massTimeConstant;      // Time constant for mass damping
    private double _viscosityTimeConstant;  // Time constant for viscosity
    private double _neighborhoodRadius;  // Neighborhood radius
    private InfluenceFunction _influenceFunction;

    this(
        double horizon,
        double massTimeConstant = double.infinity,      // Default: no mass damping
        double viscosityTimeConstant = double.infinity,            // Default: 0.1 s
        InfluenceFunction influenceFunction = &cubicSplineInfluence
    ) {
        _massTimeConstant = massTimeConstant;
        _viscosityTimeConstant = viscosityTimeConstant;
        _neighborhoodRadius = horizon;
        _influenceFunction = influenceFunction;
    }
    
    // Implement global damping (mass and stiffness)
    V calculateGlobalForce(V velocity, double mass) const {
        // Early return if mass damping is disabled (infinite time constant)
        if (_massTimeConstant == double.infinity) {
            return V.zero();
        }
        // Mass damping: F = -mv/τₘ
        return velocity * (-mass / _massTimeConstant);
    }
    
    // Implement artificial viscosity
    V calculateBondForce(
        V relativeVelocity,
        V bondVector,
        double mass
    ) const {
        // Early return if viscosity is disabled (infinite time constant)
        if (_viscosityTimeConstant == double.infinity) {
            return V.zero();
        }
        
        double isCompressing = relativeVelocity.dot(bondVector) < 0;
        if (isCompressing) return V.zero();
        
        double bondLength = bondVector.magnitude();
        
        return -relativeVelocity * mass / _viscosityTimeConstant * _influenceFunction(bondLength, _neighborhoodRadius);
    }

    // Calculate global damping dissipation using force dot velocity
    double calculateGlobalDissipation(V velocity, double mass, double timeStep) const {
        if (_massTimeConstant == double.infinity) {
            return 0.0;
        }
        return -0.5 * calculateGlobalForce(velocity, mass).dot(velocity) * timeStep;
    }
    
    // Calculate per-bond damping dissipation using force dot velocity
    double calculateBondDissipation(
        V relativeVelocity,
        V bondVector,
        double mass,
        double timeStep
    ) const {
        if (_viscosityTimeConstant == double.infinity) {
            return 0.0;
        }
        return -0.5 * calculateBondForce(relativeVelocity, bondVector, mass).dot(relativeVelocity) * timeStep;
    }
}

// Bond-based material point implementation
class BondBasedPoint(V) : MaterialPoint!(BondBasedPoint!V, V)
    if (isVector!V) {
    private V _position;
    private V _referencePosition;
    private V _velocity;
    private double _mass;
    private bool _isVelocityFixed;  // Whether velocity updates are disabled
    private V _constantForce;  // Store constant external force
    private double _timeElapsed;      // Track simulation time
    private V _targetForce;          // Final force magnitude
    private double _rampDuration;     // Time to reach target force

    // Interface implementation
    @property V position() const { return _position; }
    @property V referencePosition() const { return _referencePosition; }
    @property V velocity() const { return _velocity; }
    @property void velocity(V vel) {
        if (!_isVelocityFixed) {
            _velocity = vel;
        }
    }
    @property void position(V newPos) { _position = newPos; }
    
    // Material properties
    private double _bondStiffness;  // Bond stiffness constant
    private double _criticalStretch;  // Critical stretch for bond breaking
    private Damper!V _damper;      // Damping strategy

    // Constructor
    this(
        V refPos,
        double pointMass,
        double bondStiffness,
        double criticalStretch,
        Damper!V damper,               // Damping strategy
        V initialVelocity = V.zero(),
        bool fixedVelocity = false,
        V targetForce = V.zero(),     // Target force at end of ramp
        double rampDuration = 1e-6    // Duration of force ramp (default 1μs)
    ) {
        _referencePosition = refPos;
        _position = refPos;
        _velocity = initialVelocity;
        _mass = pointMass;
        _bondStiffness = bondStiffness;
        _criticalStretch = criticalStretch;
        _damper = damper;
        _isVelocityFixed = fixedVelocity;
        _targetForce = targetForce;
        _rampDuration = rampDuration;
        _timeElapsed = 0.0;
        _constantForce = V.zero();  // Start with zero force
    }

    // Force property accessors
    @property V targetForce() const {
        return _targetForce;
    }

    @property void targetForce(V force) {
        _targetForce = force;
    }

    @property double rampDuration() const {
        return _rampDuration;
    }

    @property void rampDuration(double duration) {
        _rampDuration = duration;
    }

    @property V currentForce() const {
        return _constantForce;
    }
    
    // Lagrangian calculation
    double computeLagrangian(const(BondBasedPoint!V)[] neighbors, V proposedPosition, double timeStep) const {
        // Kinetic energy T = (1/2)m((x2-x1)/dt)^2
        V proposedVelocity = (proposedPosition - _position) / timeStep;
        double kineticEnergy = 0.5 * _mass * proposedVelocity.magnitudeSquared();
        
        // Potential energy from bonds
        double potentialEnergy = 0.0;
        foreach (neighbor; neighbors) {
            // Calculate reference and proposed vectors
            V refVector = neighbor.referencePosition - _referencePosition;
            V propVector = neighbor._position - proposedPosition;
            
            double refLength = refVector.magnitude();
            double propLength = propVector.magnitude();
            
            // Calculate stretch
            double stretch = (propLength - refLength) / refLength;
            
            // Bond energy if not broken
            if (abs(stretch) <= _criticalStretch) {
                potentialEnergy += 0.5 * _bondStiffness * stretch * stretch;
            }
            else {
                potentialEnergy += 0.5 * _bondStiffness * _criticalStretch * _criticalStretch; // Energy at critical stretch
            }
        }
        
        // Calculate dissipation from damping
        double totalDissipation = 0.0;
        // Global damping dissipation
        totalDissipation += _damper.calculateGlobalDissipation(proposedVelocity, _mass, timeStep);
        
        // Bond damping dissipation
        foreach (neighbor; neighbors) {
            V proposedRelativeVelocity = (neighbor._velocity - proposedVelocity);
            V propVector = neighbor._position - proposedPosition;
            
            totalDissipation += _damper.calculateBondDissipation(
                proposedRelativeVelocity,
                propVector,
                _mass,
                timeStep
            );
        }
        
        // Add potential energy contribution from constant external forces
        // Note: Negative because forces point in direction of decreasing potential
        potentialEnergy -= _constantForce.dot(proposedPosition - _referencePosition);
        
        // Lagrangian = T - V - D
        return kineticEnergy - potentialEnergy - totalDissipation;
    }
    
    // Check if a bond is under compression
    private bool isCompressing(const(BondBasedPoint!V) neighbor) const {
        V relativeVelocity = neighbor.velocity - _velocity;
        V bondVector = neighbor.position - _position;
        return relativeVelocity.dot(bondVector) < 0;
    }

    // Calculate bond force between two points with reversible damage
    private V bondForce(const(BondBasedPoint!V) neighbor) const {
        // Calculate reference and current vectors between points
        V refVector = neighbor.referencePosition - _referencePosition;
        V curVector = neighbor.position - _position;
        
        double refLength = refVector.magnitude();
        double curLength = curVector.magnitude();
        enforce(curLength >= PLANCK_LENGTH, "Current bond length is shorter than Planck length");
        
        // Calculate stretch
        double stretch = (curLength - refLength) / refLength;
        
        // Reversible damage model: zero force if stretch exceeds critical value
        if (abs(stretch) > _criticalStretch) {
            return V.zero();
        }
        
        // Bond force calculation (linear micropotential)
        V elasticForce = curVector.unit() * (_bondStiffness * stretch);
        
        // Add bond damping only during compression
        if (isCompressing(neighbor)) {
            return elasticForce + _damper.calculateBondForce(
                neighbor.velocity - _velocity,
                curVector,
                _mass
            );
        }
        
        return elasticForce;
    }
    
    // Calculate total force on point
    private V calculateTotalForce(const(BondBasedPoint!V)[] neighbors) {
        // Calculate elastic force from bonds
        V elasticForce = V.zero();
        foreach (neighbor; neighbors) {
            elasticForce = elasticForce + bondForce(neighbor);
        }
        
        // Add global damping force
        V dampingForce = _damper.calculateGlobalForce(_velocity, _mass);
        
        // Update ramped force
        double rampFactor = _timeElapsed / _rampDuration;
        if (rampFactor > 1.0) rampFactor = 1.0;  // Clamp at maximum
        _constantForce = _targetForce * rampFactor;  // Linear interpolation
        
        return elasticForce + dampingForce + _constantForce;
    }
    
    // Velocity Verlet integration state update implementation
    void updateState(const(BondBasedPoint!V)[] neighbors, double timeStep) {
        if (_isVelocityFixed) {
            // For fixed velocity points, just update position
            _position = _position + _velocity * timeStep;
            return;
        }
        
        // Update time for force ramping
        _timeElapsed += timeStep;
        
        // 1. Calculate initial forces
        V initialForce = calculateTotalForce(neighbors);
        
        // 2. Update velocity by half timestep
        V halfStepVelocity = _velocity + initialForce * (timeStep * 0.5 / _mass);
        
        // 3. Update position using half-step velocity
        _position = _position + halfStepVelocity * timeStep;
        
        // 4. Calculate new forces at updated position
        V newForce = calculateTotalForce(neighbors);
        
        // 5. Complete velocity update with new forces
        _velocity = halfStepVelocity + newForce * (timeStep * 0.5 / _mass);
        enforce(velocity.magnitudeSquared() < SPEED_OF_LIGHT ^^ 2, "Velocity exceeds speed of light");
    }
}

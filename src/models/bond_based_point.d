module models.bond_based_point;

import core.material_point;
import core.damper : Damper;
import core.velocity_constraint : VelocityConstraint;
import math.vector;
import std.math : abs;
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
                potentialEnergy += -0.5 * _bondStiffness * stretch * stretch;
            }
            else {
                potentialEnergy += -0.5 * _bondStiffness * _criticalStretch * _criticalStretch; // Energy at critical stretch
            }
        }
        
        // Calculate dissipation from damping
        double totalDissipation = 0.0;
        // Global damping dissipation
        totalDissipation += _damper.calculateGlobalDissipation(proposedVelocity, _mass, timeStep);
        
        // Bond damping dissipation
        foreach (neighbor; neighbors) {
            // Calculate proposed and reference vectors for dissipation
            V propVector = neighbor._position - proposedPosition;
            V refVector = neighbor.referencePosition - _referencePosition;
            V displacement = propVector - refVector;
            
            totalDissipation += _damper.calculateBondDissipation(
                neighbor._velocity - proposedVelocity,  // Relative velocity
                displacement,  // Use displacement direction for damping
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
    
    // Check if a bond is under compression using displacement direction
    private bool isCompressing(const(BondBasedPoint!V) neighbor) const {
        V relativeVelocity = neighbor.velocity - _velocity;
        V refVector = neighbor.referencePosition - _referencePosition;
        V curVector = neighbor.position - _position;
        V displacement = curVector - refVector;
        return relativeVelocity.dot(displacement) < 0;
    }

    // Calculate bond force between two points with reversible damage
    private V bondForce(const(BondBasedPoint!V) neighbor) const {
        // Calculate reference and current vectors between points
        V refVector = neighbor.referencePosition - _referencePosition;
        V curVector = neighbor.position - _position;
        V displacement = curVector - refVector;  // Calculate displacement once for reuse
        
        double refLength = refVector.magnitude();
        double curLength = curVector.magnitude();
        enforce(curLength >= PLANCK_LENGTH, "Current bond length is shorter than Planck length");
        
        // Calculate stretch
        double stretch = (curLength - refLength) / refLength;
        
        // Reversible damage model: zero force if stretch exceeds critical value
        if (abs(stretch) > _criticalStretch) {
            return V.zero();
        }
        
        // Bond force calculation
        V forceDirection = -displacement.unit();  // Unit vector opposing displacement
        V elasticForce = forceDirection * (_bondStiffness * stretch);
        
        // Add bond damping only during compression
        if (isCompressing(neighbor)) {
            return elasticForce + _damper.calculateBondForce(
                neighbor.velocity - _velocity,
                displacement,  // Use displacement for damping direction
                _mass
            );
        }
        
        return elasticForce;
    }
}

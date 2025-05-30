module core.damper;

import math.vector;
import std.math : abs, isNaN;

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
        if (_massTimeConstant == double.infinity || isNaN(_massTimeConstant)) {
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
        if (_viscosityTimeConstant == double.infinity || isNaN(_viscosityTimeConstant)) {
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

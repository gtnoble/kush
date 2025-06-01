module core.damper;

import math.vector;
import std.math : abs, isNaN, exp;
import std.complex : exp;

// Cubic spline influence function for bond-based models
auto cubicSplineInfluence(T, U)(T bondLength, U neighborhoodRadius) {
    auto q = bondLength / neighborhoodRadius;
    return 1.0 - 3.0 * q * q + 2.0 * q * q * q;
} 

// Implementation combining exponential damping and Monaghan viscosity
class Damper(V) if (isVector!V) {
    private double _massTimeConstant;      // Time constant for mass damping
    private double _viscosityTimeConstant;  // Time constant for viscosity
    private double _neighborhoodRadius;  // Neighborhood radius

    this(
        double horizon,
        double massTimeConstant = double.infinity,      // Default: no mass damping
        double viscosityTimeConstant = double.infinity,            // Default: 0.1 s
    ) {
        _massTimeConstant = massTimeConstant;
        _viscosityTimeConstant = viscosityTimeConstant;
        _neighborhoodRadius = horizon;
    }
    
    // Implement global damping (mass and stiffness)
    U calculateGlobalForce(U)(U velocity, double mass) const {
        // Early return if mass damping is disabled (infinite time constant)
        if (_massTimeConstant == double.infinity || isNaN(_massTimeConstant)) {
            return U.zero();
        }
        // Mass damping: F = -mv/τₘ
        return velocity * (-mass / _massTimeConstant);
    }
    
    // Implement artificial viscosity
    U calculateBondForce(U)(
        U relativeVelocity,
        U bondVector,
        double mass
    ) const {
        // Early return if viscosity is disabled (infinite time constant)
        if (_viscosityTimeConstant == double.infinity || isNaN(_viscosityTimeConstant)) {
            return U.zero();
        }
        
        auto compression = relativeVelocity.dot(bondVector);
        auto alpha = 1.0 / (1.0 + exp(-2.0 * compression)); // logistic function replacement
        
        auto bondLength = bondVector.magnitude();
        
        return alpha * -relativeVelocity * mass / _viscosityTimeConstant * cubicSplineInfluence(bondLength, _neighborhoodRadius);
    }

    // Calculate global damping dissipation using force dot velocity
    T calculateGlobalDissipation(T, U)(U velocity, double mass, double timeStep) const {
        if (_massTimeConstant == double.infinity) {
            return T(0);
        }
        return T(-0.5) * calculateGlobalForce(velocity, mass).dot(velocity) * T(timeStep);
    }
    
    // Calculate per-bond damping dissipation using force dot velocity
    T calculateBondDissipation(T, U)(
        U relativeVelocity,
        U bondVector,
        double mass,
        double timeStep
    ) const {
        if (_viscosityTimeConstant == double.infinity) {
            return T(0);
        }
        return T(-0.5) * calculateBondForce(relativeVelocity, bondVector, mass).dot(relativeVelocity) * T(timeStep);
    }
}

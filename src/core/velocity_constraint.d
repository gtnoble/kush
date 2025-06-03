module core.velocity_constraint;

import math.vector;
import core.material_body;
import core.material_point;
import std.math : sqrt;

// Simple velocity constraint for a single point
struct VelocityConstraint(V) if (isVector!V) {
    private V _targetVelocity;

    this(V target) {
        _targetVelocity = target;
    }
    
    @property V targetVelocity() const { return _targetVelocity; }

    // Evaluate constraint violation: g(v) = v - v_target
    V evaluateViolation(V velocity) const {
        return velocity - _targetVelocity;
    }

    // Get constraint Jacobian with respect to position using chain rule:
    // ∂g/∂x = (∂g/∂v) * (∂v/∂x)
    // where ∂g/∂v = 1 (identity) and ∂v/∂x = 1/Δt
    double getJacobianComponent(double timeStep) const {
        // ∂g/∂v = 1 (derivative of violation with respect to velocity)
        double dgdv = 1.0;
        
        // ∂v/∂x = 1/Δt (change in velocity per change in position)
        double dvdx = 1.0 / timeStep;

        
        // Chain rule: ∂g/∂x = (∂g/∂v) * (∂v/∂x)
        return dgdv * dvdx;
    }
}

// Manages all velocity constraints in the system
class ConstraintManager(T, V) if (isMaterialPoint!(T, V)) {
    private struct ConstraintInfo {
        size_t pointIndex;
        VelocityConstraint!V constraint;
    }
    
    private ConstraintInfo[] _constraints;
    private V[] _previousPositions;
    
    this() {
        _constraints = [];
    }
    
    // Collect constraints from material body
    void collectConstraints(MaterialBody!(T, V) body) {
        _constraints.length = 0;
        _previousPositions.length = body.numPoints;
        
        // Store current positions as previous positions for next iteration
        for (size_t i = 0; i < body.numPoints; ++i) {
            _previousPositions[i] = body[i].position;
            
            if (body[i].velocityConstraint !is null) {
                _constraints ~= ConstraintInfo(i, *body[i].velocityConstraint);
            }
        }
    }
    
    // Get total number of constraint equations (each constraint contributes V.length equations)
    @property size_t numConstraintEquations() const {
        return _constraints.length * V.length;
    }
    
    // Evaluate all constraint violations
    void evaluateConstraints(const V[] positions, double timeStep, ref double[] violations) {
        violations.length = numConstraintEquations;
        
        size_t violationIndex = 0;
        foreach (const ref constraintInfo; _constraints) {
            // Compute current velocity from position change
            auto currentVelocity = (positions[constraintInfo.pointIndex] - 
                                  _previousPositions[constraintInfo.pointIndex]) / timeStep;
            
            auto violation = constraintInfo.constraint.evaluateViolation(currentVelocity);
            
            // Store each component of the violation vector
            foreach (component; 0..V.length) {
                violations[violationIndex++] = violation[component];
            }
        }
    }
    
    // Compute constraint forces: F_constraint = -λ * ∂g/∂x
    void computeConstraintForces(const DynamicVector!double multipliers, double timeStep, ref V[] forces) {
        // Initialize forces to zero
        foreach (ref force; forces) {
            force = V.zero();
        }
        
        size_t multiplierIndex = 0;
        foreach (const ref constraintInfo; _constraints) {
            auto jacobianComponent = constraintInfo.constraint.getJacobianComponent(timeStep);
            
            // Apply constraint force for each component
            foreach (component; 0..V.length) {
                forces[constraintInfo.pointIndex][component] -= 
                    multipliers[multiplierIndex++] * jacobianComponent;
            }
        }
    }
    
    // Get constraint violation for a specific constraint equation (used in gradient computation)
    double getConstraintViolation(size_t constraintEquationIndex, const V[] positions, double timeStep) const {
        size_t constraintIndex = constraintEquationIndex / V.length;
        size_t componentIndex = constraintEquationIndex % V.length;
        
        if (constraintIndex >= _constraints.length) {
            return 0.0;
        }
        
        const constraintInfo = _constraints[constraintIndex];
        
        // Compute current velocity from position change
        auto currentVelocity = (positions[constraintInfo.pointIndex] - 
                              _previousPositions[constraintInfo.pointIndex]) / timeStep;
        
        auto violation = constraintInfo.constraint.evaluateViolation(currentVelocity);
        
        return violation[componentIndex];
    }
}

// State vector management for Lagrangian system
struct LagrangianState(V) if (isVector!V) {
    DynamicVector!double positions;
    DynamicVector!double multipliers;
    
    // Create from unified state vector
    static LagrangianState fromUnified(const DynamicVector!double unified, size_t numPoints) {
        LagrangianState state;
        size_t positionComponents = numPoints * V.length;
        
        state.positions = DynamicVector!double(positionComponents);
        state.multipliers = DynamicVector!double(unified.length - positionComponents);
        
        // Extract positions
        foreach (i; 0..positionComponents) {
            state.positions[i] = unified[i];
        }
        
        // Extract multipliers
        foreach (i; 0..state.multipliers.length) {
            state.multipliers[i] = unified[positionComponents + i];
        }
        
        return state;
    }
    
    // Convert to unified state vector
    DynamicVector!double toUnified() const {
        auto unified = DynamicVector!double(positions.length + multipliers.length);
        
        // Copy positions
        foreach (i; 0..positions.length) {
            unified[i] = positions[i];
        }
        
        // Copy multipliers
        foreach (i; 0..multipliers.length) {
            unified[positions.length + i] = multipliers[i];
        }
        
        return unified;
    }
    
    // Extract position vectors from state
    V[] extractPositions(size_t numPoints) const {
        auto result = new V[numPoints];
        foreach (i; 0..numPoints) {
            foreach (j; 0..V.length) {
                result[i][j] = positions[i * V.length + j];
            }
        }
        return result;
    }
    
    // Update positions in state
    void updatePositions(const V[] newPositions) {
        foreach (i; 0..newPositions.length) {
            foreach (j; 0..V.length) {
                positions[i * V.length + j] = newPositions[i][j];
            }
        }
    }
}

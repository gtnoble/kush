module core.integration;

import core.material_body;
import core.material_point;
import core.optimization : OptimizationState, ObjectiveFunction, PartialDerivativeFunction, DIFFERENCE_COEFFS, Minimizer, GradientMinimizer;
import math.vector;
import std.math : abs, asinh, sqrt;
import std.algorithm : max;

import core.velocity_constraint : VelocityConstraint, SystemVelocityConstraint, ConstraintTerm;

// Enum for derivative computation method
enum DerivativeMethod {
    Analytical,    // Use analytical gradients for positions
    FiniteDifference // Use finite differences for positions
}

// Create a system Lagrangian objective function
ObjectiveFunction!V createSystemLagrangian(T, V)(MaterialBody!(T, V) body, double timeStep) if (isVector!V) {
    // Capture state in closure
    auto currentPositions = new V[body.numPoints];
    auto proposedVelocities = new V[body.numPoints];
    
    // Initialize current positions
    for (size_t i = 0; i < body.numPoints; ++i) {
        currentPositions[i] = body[i].position;
    }

    // Get constraint terms
    auto constraints = SystemVelocityConstraint!(T,V).getSystemConstraints(body);
    
    // Return objective function delegate
    return (const OptimizationState!V state) {
        double totalLagrangian = 0.0;
        
        T[] neighbors;
        // Calculate proposed velocities from state
        for (size_t i = 0; i < body.numPoints; ++i) {
            proposedVelocities[i] = (state.positions[i] - currentPositions[i]) / timeStep;
        }
        
        // For each point, compute its contribution
        for (size_t i = 0; i < body.numPoints; ++i) {
            auto point = body[i];
            body.neighbors(i, neighbors);
            
            // Compute regular Lagrangian
            totalLagrangian += point.computeLagrangian(
                neighbors, 
                state.positions[i], 
                timeStep
            );
        }
        
        // Add contributions from each component-wise constraint with its multiplier
        for (size_t i = 0; i < constraints.length; ++i) {
            auto constraint = constraints[i];
            double violation = constraint.evaluate(proposedVelocities[constraint.pointIndex]);
            // Note: Each component's constraint contributes linearly to the Lagrangian
            totalLagrangian -= state.multipliers[i] * violation;
        }
        
        return totalLagrangian;
    };
}

// Create a system Lagrangian gradient function
PartialDerivativeFunction!V createSystemLagrangianPartialDerivative(T, V)(
    MaterialBody!(T, V) body, 
    double timeStep,
    DerivativeMethod positionDerivativeMethod = DerivativeMethod.Analytical,
    int finiteDifferenceOrder = 2
) if (isVector!V) {
    // Capture state in closure
    auto currentPositions = new V[body.numPoints];
    
    // Initialize current positions
    for (size_t i = 0; i < body.numPoints; ++i) {
        currentPositions[i] = body[i].position;
    }

    // Get constraint terms
    auto constraints = SystemVelocityConstraint!(T,V).getSystemConstraints(body);
    
    // Conditionally capture the objective function for finite difference
    ObjectiveFunction!V objective;
    if (positionDerivativeMethod == DerivativeMethod.FiniteDifference) {
        objective = createSystemLagrangian!(T, V)(body, timeStep);
    }
    
    // Return gradient function delegate
    return (const OptimizationState!V state, size_t componentIndex) {
        T[] neighbors;
        
        // Handle position components
        if (componentIndex < body.numPoints * V.length) {
            size_t pointIndex = componentIndex / V.length;
            size_t dimIndex = componentIndex % V.length;
            
            if (positionDerivativeMethod == DerivativeMethod.Analytical) {
                // Get neighbors for this point
                body.neighbors(pointIndex, neighbors);
                
                // Compute gradient using analytical method
                V grad = body[pointIndex].computeLagrangianGradient(
                    neighbors,
                    state.positions[pointIndex],
                    timeStep
                );
                
                // Return the specific dimension's derivative
                return grad[dimIndex];
            } else {
                // Finite difference method
                auto coeffs = DIFFERENCE_COEFFS[finiteDifferenceOrder];
                int stencilLength = cast(int)coeffs.length;
                int halfPoints = (stencilLength - 1) / 2;
                
                double step_size = sqrt(double.epsilon) * max(abs(state[componentIndex]), 1);
                double derivative = 0.0;
                
                for (int j = 0; j < stencilLength; j++) {
                    if (coeffs[j] == 0.0) continue;
                    
                    auto eval_state = state.dup();
                    eval_state[componentIndex] += step_size * (j - halfPoints);
                    double eval = objective(eval_state);
                    derivative += coeffs[j] * eval;
                }
                
                return derivative / step_size;
            }
        }
        // Handle multiplier components
        else {
            size_t constraintIndex = componentIndex - body.numPoints * V.length;
            auto constraint = constraints[constraintIndex];
            
            // Compute only the needed velocity for constraint evaluation
            auto velocity = (state.positions[constraint.pointIndex] - currentPositions[constraint.pointIndex]) / timeStep;
            
            // Return negative constraint violation (for Lagrangian derivative)
            return -constraint.evaluate(velocity);
        }
    };
}

// Lagrangian integration strategy
// Core integration class that uses optimization to solve the Lagrangian equations of motion
class LagrangianIntegrator(T, V) {
    private:
        GradientMinimizer!V _solver;
        string _derivativeMethod;
        int _finiteDifferenceOrder;
        
        public:
            this(GradientMinimizer!V solver, 
                 string derivativeMethod = "analytical",
                 int finiteDifferenceOrder = 2) 
        {
            _solver = solver;
            _derivativeMethod = derivativeMethod;
            _finiteDifferenceOrder = finiteDifferenceOrder;
        }
        
        void integrate(MaterialBody!(T, V) body, double timeStep) {
            // Create gradient function with configured method
            auto gradientFunction = createSystemLagrangianPartialDerivative!T(
                body, 
                timeStep,
                (_derivativeMethod == "finite_difference") 
                    ? DerivativeMethod.FiniteDifference 
                    : DerivativeMethod.Analytical,
                _finiteDifferenceOrder
            );
            
            // Create objective function
            auto objective = createSystemLagrangian!(T, V)(body, timeStep);
            
            // Get current positions
            V[] currentPositions = new V[body.numPoints];
            
            // Initialize current state
            for (size_t i = 0; i < body.numPoints; ++i) {
                auto point = body[i];
                currentPositions[i] = point.position;
            }
            
            // Get the number of constraints for initialization
            auto constraints = SystemVelocityConstraint!(T,V).getSystemConstraints(body);
            
            // Create initial state with positions and zero multipliers
            auto initialState = OptimizationState!V(currentPositions, DynamicVector!double.zero(constraints.length));
            
            // Call minimize with unified state
            auto result = _solver(
                initialState,
                objective,
                gradientFunction
            );
            
            // Update positions and velocities with optimized values
            for (size_t i = 0; i < body.numPoints; ++i) {
                auto point = body[i];
                V oldPosition = point.position;
                point.position = result.positions[i];
                point.velocity = (result.positions[i] - oldPosition) / timeStep;
            }
        }
}

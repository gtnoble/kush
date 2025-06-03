module core.integration;

import core.material_body;
import core.material_point;
import core.optimization : ObjectiveFunction, PartialDerivativeFunction, DIFFERENCE_COEFFS, Minimizer, GradientMinimizer;
import math.vector;
import std.math : abs, asinh, sqrt;
import std.algorithm : max;

import core.velocity_constraint : VelocityConstraint, ConstraintManager, LagrangianState;

// Enum for derivative computation method
enum DerivativeMethod {
    Analytical,    // Use analytical gradients for positions
    FiniteDifference // Use finite differences for positions
}

// Create a system Lagrangian objective function
ObjectiveFunction createSystemLagrangian(T, V)(MaterialBody!(T, V) body, double timeStep) if (isVector!V) {
    // Create constraint manager
    auto constraintManager = new ConstraintManager!(T, V)();
    constraintManager.collectConstraints(body);
    
    // Return objective function delegate
    return (const DynamicVector!double unifiedState) {
        double totalLagrangian = 0.0;
        
        T[] neighbors;
        
        // Parse state using LagrangianState
        auto state = LagrangianState!V.fromUnified(unifiedState, body.numPoints);
        auto positions = state.extractPositions(body.numPoints);
        
        // For each point, compute its contribution to the Lagrangian
        for (size_t i = 0; i < body.numPoints; ++i) {
            auto point = body[i];
            body.neighbors(i, neighbors);
            
            // Compute regular Lagrangian (kinetic - potential - dissipation)
            totalLagrangian += point.computeLagrangian(
                neighbors, 
                positions[i], 
                timeStep
            );
        }
        
        // Add constraint contributions: -λ * g(x)
        double[] violations;
        constraintManager.evaluateConstraints(positions, timeStep, violations);
        
        for (size_t i = 0; i < violations.length; ++i) {
            totalLagrangian -= state.multipliers[i] * violations[i];
        }
        
        return totalLagrangian;
    };
}

// Create a system Lagrangian gradient function
PartialDerivativeFunction createSystemLagrangianPartialDerivative(T, V)(
    MaterialBody!(T, V) body, 
    double timeStep,
    DerivativeMethod positionDerivativeMethod = DerivativeMethod.Analytical,
    int finiteDifferenceOrder = 2
) if (isVector!V) {
    // Create constraint manager
    auto constraintManager = new ConstraintManager!(T, V)();
    constraintManager.collectConstraints(body);
    
    // Conditionally capture the objective function for finite difference
    ObjectiveFunction objective;
    if (positionDerivativeMethod == DerivativeMethod.FiniteDifference) {
        objective = createSystemLagrangian!(T, V)(body, timeStep);
    }
    
    // Return gradient function delegate
    return (const DynamicVector!double unifiedState, size_t componentIndex) {
        T[] neighbors;
        
        // Parse state using LagrangianState
        auto state = LagrangianState!V.fromUnified(unifiedState, body.numPoints);
        size_t numPositionComponents = body.numPoints * V.length;
        
        // Handle position components
        if (componentIndex < numPositionComponents) {
            size_t pointIndex = componentIndex / V.length;
            size_t dimIndex = componentIndex % V.length;
            
            if (positionDerivativeMethod == DerivativeMethod.Analytical) {
                auto positions = state.extractPositions(body.numPoints);
                
                // Get neighbors for this point
                body.neighbors(pointIndex, neighbors);
                
                // Compute physics gradient using analytical method
                V physicsGrad = body[pointIndex].computeLagrangianGradient(
                    neighbors,
                    positions[pointIndex],
                    timeStep
                );
                
                // Compute constraint forces
                V[] constraintForces = new V[body.numPoints];
                constraintManager.computeConstraintForces(
                    state.multipliers, timeStep, constraintForces);
                
                // Total gradient = physics gradient + constraint forces
                return physicsGrad[dimIndex] + constraintForces[pointIndex][dimIndex];
            } else {
                // Finite difference method
                auto coeffs = DIFFERENCE_COEFFS[finiteDifferenceOrder];
                int stencilLength = cast(int)coeffs.length;
                int halfPoints = (stencilLength - 1) / 2;
                
                double step_size = sqrt(double.epsilon) * max(abs(unifiedState[componentIndex]), 1);
                double derivative = 0.0;
                
                for (int j = 0; j < stencilLength; j++) {
                    if (coeffs[j] == 0.0) continue;
                    
                    auto eval_state = unifiedState.dup;
                    eval_state[componentIndex] += step_size * (j - halfPoints);
                    double eval = objective(eval_state);
                    derivative += coeffs[j] * eval;
                }
                
                return derivative / step_size;
            }
        }
        // Handle multiplier components
        else {
            size_t constraintIndex = componentIndex - numPositionComponents;
            auto positions = state.extractPositions(body.numPoints);
            
            // Return negative constraint violation (for Lagrangian derivative w.r.t. multiplier)
            return -constraintManager.getConstraintViolation(constraintIndex, positions, timeStep);
        }
    };
}

// Lagrangian integration strategy
// Core integration class that uses optimization to solve the Lagrangian equations of motion
class LagrangianIntegrator(T, V) {
    private:
        GradientMinimizer _solver;
        string _derivativeMethod;
        int _finiteDifferenceOrder;
        
        public:
            this(GradientMinimizer solver, 
                 string derivativeMethod = "analytical",
                 int finiteDifferenceOrder = 2) 
        {
            _solver = solver;
            _derivativeMethod = derivativeMethod;
            _finiteDifferenceOrder = finiteDifferenceOrder;
        }
        
        void integrate(MaterialBody!(T, V) body, double timeStep) {
            // Create constraint manager and collect constraints
            auto constraintManager = new ConstraintManager!(T, V)();
            constraintManager.collectConstraints(body);
            
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
            for (size_t i = 0; i < body.numPoints; ++i) {
                currentPositions[i] = body[i].position;
            }
            
            // Create initial state using LagrangianState
            auto initialState = LagrangianState!V();
            initialState.positions = DynamicVector!double(body.numPoints * V.length);
            initialState.multipliers = DynamicVector!double.zero(constraintManager.numConstraintEquations);
            
            // Initialize positions in state
            initialState.updatePositions(currentPositions);
            
            // Convert to unified state for optimization
            auto initialUnified = initialState.toUnified();
            
            // Call minimize with unified state
            auto resultUnified = _solver(
                initialUnified,
                objective,
                gradientFunction
            );
            
            // Parse result and update body
            auto resultState = LagrangianState!V.fromUnified(resultUnified, body.numPoints);
            auto optimizedPositions = resultState.extractPositions(body.numPoints);
            
            for (size_t i = 0; i < body.numPoints; ++i) {
                auto point = body[i];
                V oldPosition = point.position;
                point.position = optimizedPositions[i];
                point.velocity = (optimizedPositions[i] - oldPosition) / timeStep;
            }
        }
}

// Unit tests for LagrangianState
unittest {
    import std.stdio : writeln;
    import std.exception : assertThrown;
    import std.math : isClose;

    // Helper function to compare vectors
    bool vectorsEqual(T)(const T[] a, const T[] b) {
        if (a.length != b.length) return false;
        foreach (i; 0..a.length) {
            if (!a[i].opEquals(b[i])) return false;
        }
        return true;
    }

    // Test LagrangianState with 2D vectors
    {
        writeln("Testing LagrangianState with 2D vectors...");

        auto positions2D = [
            Vector!(double, 2)([1.0, 2.0]),
            Vector!(double, 2)([3.0, 4.0])
        ];
        
        auto state = LagrangianState!(Vector!(double, 2))();
        state.positions = DynamicVector!double(4);
        state.multipliers = DynamicVector!double([5.0, 6.0]);
        
        state.updatePositions(positions2D);
        auto unified = state.toUnified();
        
        assert(unified.length == 6); // 4 position components + 2 multipliers
        assert(unified[0] == 1.0);
        assert(unified[1] == 2.0);
        assert(unified[2] == 3.0);
        assert(unified[3] == 4.0);
        assert(unified[4] == 5.0);
        assert(unified[5] == 6.0);

        // Test round-trip conversion
        auto reconstructed = LagrangianState!(Vector!(double, 2)).fromUnified(unified, 2);
        auto extractedPositions = reconstructed.extractPositions(2);
        
        assert(vectorsEqual(extractedPositions, positions2D));
        assert(reconstructed.multipliers[0] == 5.0);
        assert(reconstructed.multipliers[1] == 6.0);
    }

    // Test LagrangianState with 3D vectors
    {
        writeln("Testing LagrangianState with 3D vectors...");

        auto positions3D = [
            Vector!(double, 3)([1.0, 2.0, 3.0]),
            Vector!(double, 3)([4.0, 5.0, 6.0])
        ];
        
        auto state = LagrangianState!(Vector!(double, 3))();
        state.positions = DynamicVector!double(6);
        state.multipliers = DynamicVector!double([7.0, 8.0]);
        
        state.updatePositions(positions3D);
        auto unified = state.toUnified();
        
        assert(unified.length == 8); // 6 position components + 2 multipliers
        assert(unified[0] == 1.0);
        assert(unified[1] == 2.0);
        assert(unified[2] == 3.0);
        assert(unified[3] == 4.0);
        assert(unified[4] == 5.0);
        assert(unified[5] == 6.0);
        assert(unified[6] == 7.0);
        assert(unified[7] == 8.0);

        // Test round-trip conversion
        auto reconstructed = LagrangianState!(Vector!(double, 3)).fromUnified(unified, 2);
        auto extractedPositions = reconstructed.extractPositions(2);
        
        assert(vectorsEqual(extractedPositions, positions3D));
        assert(reconstructed.multipliers[0] == 7.0);
        assert(reconstructed.multipliers[1] == 8.0);
    }
}

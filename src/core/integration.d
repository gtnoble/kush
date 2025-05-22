module core.integration;

import core.material_body;
import core.material_point;
import core.optimization;
import math.vector;
import std.math : abs;

import core.velocity_constraint : VelocityConstraint, SystemVelocityConstraint, ConstraintTerm;

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

// Lagrangian integration strategy
// Core integration class that uses optimization to solve the Lagrangian equations of motion
class LagrangianIntegrator(T, V) {
    private:
        Minimizer!V _solver;
        
    public:
        this(Minimizer!V solver) {
            _solver = solver;
        }
        
        void integrate(MaterialBody!(T, V) body, double timeStep) {
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
            auto initialState = OptimizationState!V(currentPositions, DynamicVector.zero(constraints.length));
            
            // Call minimize with unified state
            auto result = _solver(
                initialState,
                objective
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

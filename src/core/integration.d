module core.integration;

import core.material_body;
import core.material_point;
import core.optimization;
import math.vector;
import std.math : abs;

import core.velocity_constraint : VelocityConstraint, SystemVelocityConstraint, ConstraintTerm;

// System Lagrangian implementation
class SystemLagrangian(T, V) : ObjectiveFunction!(T, V) if (isMaterialPoint!(T, V)) {
    private:
        MaterialBody!(T, V) _body;
        double _timeStep;
        V[] _currentPositions;
        V[] _proposedVelocities;  // Cache for proposed velocities
        ConstraintTerm!V[] _constraints;  // Array of individual constraint terms
        
    public:
        this(MaterialBody!(T, V) body, double timeStep) {
            _body = body;
            _timeStep = timeStep;
            
            // Store current positions for velocity calculations
            _currentPositions = new V[body.numPoints];
            _proposedVelocities = new V[body.numPoints];
            for (size_t i = 0; i < body.numPoints; ++i) {
                _currentPositions[i] = body[i].position;
            }

            // Get individual constraint terms
            _constraints = SystemVelocityConstraint!(T,V).getSystemConstraints(body);
        }
        
        override double evaluate(OptimizationState!V state) {
            double totalLagrangian = 0.0;
            
            T[] neighbors;
            // Calculate proposed velocities from state
            for (size_t i = 0; i < _body.numPoints; ++i) {
                _proposedVelocities[i] = (state.positions[i] - _currentPositions[i]) / _timeStep;
            }
            
            // For each point, compute its contribution
            for (size_t i = 0; i < _body.numPoints; ++i) {
                auto point = _body[i];
                _body.neighbors(i, neighbors);
                
                // Compute regular Lagrangian
                totalLagrangian += point.computeLagrangian(
                    neighbors, 
                    state.positions[i], 
                    _timeStep
                );
            }
            
            // Add contributions from each constraint with its own multiplier
            for (size_t i = 0; i < _constraints.length; ++i) {
                double violation = _constraints[i].evaluate(_proposedVelocities[_constraints[i].pointIndex]);
                totalLagrangian -= state.multipliers[i] * violation;
            }
            
            return totalLagrangian;
        }
}

// Lagrangian integration strategy
// Core integration class that uses optimization to solve the Lagrangian equations of motion
class LagrangianIntegrator(T, V) {
    private:
        OptimizationSolver!(T, V) _solver;
        
    public:
        this(OptimizationSolver!(T, V) solver) {
            _solver = solver;
        }
        
        void integrate(MaterialBody!(T, V) body, double timeStep) {
            // Create objective function
            auto objective = new SystemLagrangian!(T, V)(body, timeStep);
            
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
            auto result = _solver.minimize(
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

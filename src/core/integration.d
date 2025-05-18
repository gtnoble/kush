module core.integration;

import core.material_body;
import core.material_point;
import core.optimization;
import math.vector;
import std.math : abs;

import core.velocity_constraint : VelocityConstraint, SystemVelocityConstraint;

// System Lagrangian implementation
class SystemLagrangian(T, V) : ObjectiveFunction!(T, V) if (isMaterialPoint!(T, V)) {
    private:
        MaterialBody!(T, V) _body;
        double _timeStep;
        V[] _currentPositions;
        V[] _proposedVelocities;  // Cache for proposed velocities
        
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
        }
        
        double evaluate(V[] proposedPositions, double multiplier = 0.0) {
            double totalLagrangian = 0.0;
            
            T[] neighbors;
            // Calculate proposed velocities
            for (size_t i = 0; i < _body.numPoints; ++i) {
                _proposedVelocities[i] = (proposedPositions[i] - _currentPositions[i]) / _timeStep;
            }
            
            // For each point, compute its contribution to the system Lagrangian
            for (size_t i = 0; i < _body.numPoints; ++i) {
                auto point = _body[i];
                _body.neighbors(i, neighbors);
                
                // Compute regular Lagrangian
                totalLagrangian += point.computeLagrangian(
                    neighbors, 
                    proposedPositions[i], 
                    _timeStep
                );
            }
            
            // Add system-wide velocity constraint contribution
            double constraintViolation = SystemVelocityConstraint!(T,V).evaluateSystemConstraint(
                _body,
                _proposedVelocities
            );
            
            return totalLagrangian - multiplier * constraintViolation;
            //return totalLagrangian + 1e12 * constraintViolation;
            //return totalLagrangian;
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
            
            // Optimize the system with initial multiplier of 0
            // Calculate average mass for the body
            double totalMass = 0.0;
            size_t count = 0;
            for (size_t i = 0; i < body.numPoints; ++i) {
                auto point = body[i];
                totalMass += point.mass;
                count++;
            }
            double averageMass = totalMass / count;
            
            auto result = _solver.minimize(
                currentPositions,
                averageMass,  // Use average mass as multiplier
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

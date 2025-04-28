module models.bond_based_point;

import core.material_point;
import math.vector;
import std.math : abs;
import std.typecons : Nullable;

// Bond-based material point implementation
class BondBasedPoint(V) : MaterialPoint!(BondBasedPoint!V, V)
    if (isVector!V) {
    private V _position;
    private V _referencePosition;
    private V _velocity;
    private bool _isVelocityFixed;  // Whether velocity updates are disabled
    private V _constantForce;  // Store constant external force
    private double _timeElapsed;      // Track simulation time
    private V _targetForce;          // Final force magnitude
    private double _rampDuration;     // Time to reach target force

    // Velocity accessor
    @property V velocity() const {
        return _velocity;
    }

    // Velocity setter
    @property void velocity(V vel) {
        if (!_isVelocityFixed) {
            _velocity = vel;
        }
    }
    private double _mass;
    
    // Material properties
    private double _bondStiffness;  // Bond stiffness constant
    private double _criticalStretch;  // Critical stretch for bond breaking
    private double _dampingTimeConstant;  // Damping time constant (τ)
    private double _soundSpeed;      // Material sound speed
    private double _viscosityLinear; // Linear viscosity coefficient (c₀)
    private double _viscosityQuad;   // Quadratic viscosity coefficient (c₁)
    
    // Constructor
    this(
        V refPos,
        double pointMass,
        double bondStiffness,
        double criticalStretch,
        double dampingTimeConstant = double.infinity,  // Default: no damping
        V initialVelocity = V.zero(),
        bool fixedVelocity = false,
        V targetForce = V.zero(),     // Target force at end of ramp
        double rampDuration = 1e-6,   // Duration of force ramp (default 1μs)
        double soundSpeed = 1000.0,    // Default sound speed (m/s)
        double viscosityLinear = 0,  // Default linear viscosity coefficient
        double viscosityQuad = 0     // Default quadratic viscosity coefficient
    ) {
        _referencePosition = refPos;
        _position = refPos;
        _velocity = initialVelocity;
        _mass = pointMass;
        _bondStiffness = bondStiffness;
        _criticalStretch = criticalStretch;
        _dampingTimeConstant = dampingTimeConstant;
        _isVelocityFixed = fixedVelocity;
        _targetForce = targetForce;
        _rampDuration = rampDuration;
        _timeElapsed = 0.0;
        _constantForce = V.zero();  // Start with zero force
        _soundSpeed = soundSpeed;
        _viscosityLinear = viscosityLinear;
        _viscosityQuad = viscosityQuad;
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
    
    // Position properties
    @property V position() const {
        return _position;
    }
    
    @property V referencePosition() const {
        return _referencePosition;
    }
    
    // Check if a bond is under compression
    private bool isCompressing(const(BondBasedPoint!V) neighbor) const {
        V relativeVelocity = neighbor.velocity - _velocity;
        V bondVector = neighbor.position - _position;
        return relativeVelocity.dot(bondVector.unit()) < 0;
    }

    // Calculate artificial viscosity term
    private V calculateViscosity(const(BondBasedPoint!V) neighbor) const {
        V relativeVelocity = neighbor.velocity - _velocity;
        V bondVector = neighbor.position - _position;
        V bondUnit = bondVector.unit();
        
        double compressionRate = -(relativeVelocity.dot(bondUnit));
        if (compressionRate <= 0) return V.zero();
        
        double bondLength = bondVector.magnitude();
        double density = _mass / (bondLength * bondLength * bondLength);
        double linear = _viscosityLinear * density * _soundSpeed * compressionRate;
        double quad = _viscosityQuad * density * compressionRate * compressionRate;
        
        return bondUnit * (linear + quad);
    }

    // Calculate bond force between two points with reversible damage
    private V bondForce(const(BondBasedPoint!V) neighbor) const {
        // Calculate reference and current vectors between points
        V refVector = neighbor.referencePosition - _referencePosition;
        V curVector = neighbor.position - _position;
        
        double refLength = refVector.magnitude();
        double curLength = curVector.magnitude();
        
        // Calculate stretch
        double stretch = (curLength - refLength) / refLength;
        
        // Reversible damage model: zero force if stretch exceeds critical value
        if (abs(stretch) > _criticalStretch) {
            return V.zero();
        }
        
        // Bond force calculation (linear micropotential)
        V elasticForce = curVector.unit() * (_bondStiffness * stretch);
        
        // Add artificial viscosity only during compression
        if (isCompressing(neighbor)) {
            return elasticForce + calculateViscosity(neighbor);
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
        
        // Add damping force (F = -mv/τ)
        V dampingForce = _velocity * (-_mass / _dampingTimeConstant);
        
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
    }
}

// Unit tests
unittest {
    import std.math : abs, PI;
    import std.conv : to;
    import std.format : format;
    
    // Test Verlet integration with 1D harmonic oscillator
    {
        // Setup system parameters
        double mass = 1.0;
        double bondStiffness = 1.0;
        double criticalStretch = 0.5;  // Large enough to avoid breaking
        double timeStep = 0.1;
        double initialStretch = 0.1;  // 10% initial stretch
        
        // Create points with initial displacement from equilibrium
        auto p1 = new BondBasedPoint!Vector1D(
            Vector1D(0.0),
            mass,
            bondStiffness,
            criticalStretch,
            double.infinity,  // No damping
            Vector1D(0.0)    // Start at rest
        );
        auto p2 = new BondBasedPoint!Vector1D(
            Vector1D(1.1),   // Reference position at 1.0 + 0.1 stretch
            mass,
            bondStiffness,
            criticalStretch
        );
        
        // Calculate expected initial force (F = -kx)
        double expectedForce = -bondStiffness * initialStretch;
        
        // First update
        p1.updateState([p2], timeStep);
        
        // Calculate and verify initial spring force
        auto force = p1.bondForce(p2);
        assert(abs(force[0] - expectedForce) < 1e-10,
            "Initial spring force incorrect. Expected: %g, Got: %g"
            .format(expectedForce, force[0]));
        
        // Calculate expected position and velocity after velocity Verlet step
        // First half velocity update: v(t + dt/2) = v(t) + (F(t)/m)(dt/2)
        double halfStepVel = 0.0 + (expectedForce/mass) * timeStep * 0.5;
        // Position update: x(t + dt) = x(t) + v(t + dt/2)dt
        double expectedPos = 0.0 + halfStepVel * timeStep;
        // Second half velocity update with same force (since displacement is small)
        double expectedVel = halfStepVel + (expectedForce/mass) * timeStep * 0.5;
        
        // Verify position and velocity updates
        assert(abs(p1.position[0] - expectedPos) < 1e-10,
            "Position after velocity Verlet step incorrect. Expected: %g, Got: %g"
            .format(expectedPos, p1.position[0]));
            
        assert(abs(p1.velocity[0] - expectedVel) < 1e-10,
            "Velocity after velocity Verlet step incorrect. Expected: %g, Got: %g"
            .format(expectedVel, p1.velocity[0]));
        
        // Verify the velocity is non-zero (system is moving)
        assert(abs(p1.velocity[0]) > 1e-10,
            "Expected non-zero velocity in oscillating system. Got: %g"
            .format(p1.velocity[0]));
            
        // Calculate initial energy
        double springPotential = 0.5 * bondStiffness * initialStretch * initialStretch;
        double kineticEnergy = 0.5 * mass * (p1.velocity[0] * p1.velocity[0]);
        double initialEnergy = springPotential + kineticEnergy;
        
        // Run several steps and check energy conservation with velocity Verlet
        for (int i = 0; i < 10; i++) {  // Run more steps to verify stability
            p1.updateState([p2], timeStep);
            
            // Calculate current energy
            double stretch = (p2.position[0] - p1.position[0] - 1.0);
            double currentPotential = 0.5 * bondStiffness * stretch * stretch;
            double currentKinetic = 0.5 * mass * (p1.velocity[0] * p1.velocity[0]);
            double currentTotal = currentPotential + currentKinetic;
            
            // Velocity Verlet should provide better energy conservation
            assert(abs(currentTotal - initialEnergy) < 1e-8,
                "Energy not conserved in velocity Verlet integration. Expected total energy: %g, Got: %g"
                .format(initialEnergy, currentTotal));
            // Test artificial viscosity behavior
    {
        double mass = 1.0;
        double bondStiffness = 1000.0;
        double criticalStretch = 0.5;
        double timeStep = 0.1;
        double soundSpeed = 1000.0;  // 1000 m/s sound speed
        double viscosityLinear = 0.5;
        double viscosityQuad = 0.5;
        
        // Create two points moving towards each other
        auto p1 = new BondBasedPoint!Vector1D(
            Vector1D(0.0),    // At origin
            mass,
            bondStiffness,
            criticalStretch,
            double.infinity,  // No damping
            Vector1D(1.0),   // Moving right
            false,          // Not fixed
            Vector1D(0.0),  // No constant force
            1e-6,          // Default ramp duration
            soundSpeed,
            viscosityLinear,
            viscosityQuad
        );
        
        auto p2 = new BondBasedPoint!Vector1D(
            Vector1D(1.0),    // 1 unit away
            mass,
            bondStiffness,
            criticalStretch,
            double.infinity,  // No damping
            Vector1D(-1.0),  // Moving left
            false,          // Not fixed
            Vector1D(0.0),  // No constant force
            1e-6,          // Default ramp duration
            soundSpeed,
            viscosityLinear,
            viscosityQuad
        );
        
        // Calculate initial force including viscosity
        auto initialForce = p1.bondForce(p2);
        
        // Should have viscosity since points are approaching
        assert(p1.isCompressing(p2),
            "Points moving towards each other should be detected as compressing");
            
        // Calculate expected viscous force components
        double relativeSpeed = 2.0;  // Closing speed
        double bondLength = 1.0;     // Initial separation
        double density = mass / (bondLength * bondLength * bondLength);
        double expectedLinear = viscosityLinear * density * soundSpeed * relativeSpeed;
        double expectedQuad = viscosityQuad * density * relativeSpeed * relativeSpeed;
        double expectedViscousForce = expectedLinear + expectedQuad;
        
        // The total force should be larger than elastic force due to viscosity
        auto elasticOnlyForce = p1.bondForce(p2);
        assert(abs(initialForce[0]) > abs(elasticOnlyForce[0]),
            "Total force should be larger than elastic force due to viscosity");
        
        // Now move points apart and verify no viscosity
        p1.velocity = Vector1D(-1.0);  // Moving left
        p2.velocity = Vector1D(1.0);   // Moving right
        
        assert(!p1.isCompressing(p2),
            "Points moving away from each other should not be detected as compressing");
            
        // Force should now only include elastic component
        auto forceMovingApart = p1.bondForce(p2);
        assert(abs(forceMovingApart[0] - elasticOnlyForce[0]) < 1e-10,
            "Force should only include elastic component when points are moving apart");
    }
}
    }
    
    // Test 2D bond breaking
    {
        double mass = 1.0;
        double bondStiffness = 1000.0;
        double criticalStretch = 0.1;
        
        // Create two points initially at rest
        auto p1 = new BondBasedPoint!Vector2D(
            Vector2D(0.0, 0.0),
            mass,
            bondStiffness,
            criticalStretch
        );
        auto p2 = new BondBasedPoint!Vector2D(
            Vector2D(1.0, 0.0),
            mass,
            bondStiffness,
            criticalStretch
        );
        
        // Move second point to create stretch > critical
        p2._position = Vector2D(1.2, 0.0);  // 20% stretch
        
        // Update state for p1
        p1.updateState([p2], 0.1);
        
        // Force and resulting changes should be zero due to damage
        assert(abs(p1.velocity[0]) < 1e-10, 
            "Expected X velocity magnitude < 1e-10 after bond break, but got: " ~ p1.velocity[0].to!string);
        assert(abs(p1.velocity[1]) < 1e-10, 
            "Expected Y velocity magnitude < 1e-10 after bond break, but got: " ~ p1.velocity[1].to!string);
    }

    // Test damping behavior
    {
        double mass = 2.0;
        double bondStiffness = 1.0;
        double criticalStretch = 0.1;
        double dampingTimeConstant = 0.5;  // 0.5 second damping time constant
        
        // Create point with initial velocity
        auto p = new BondBasedPoint!Vector1D(
            Vector1D(0.0),
            mass,
            bondStiffness,
            criticalStretch,
            dampingTimeConstant,
            Vector1D(1.0)  // Initial velocity
        );
        
        // Update state with no neighbors (testing pure damping)
        p.updateState([], 0.1);  // 0.1s timestep
        
        // First-order decay for dt=0.1s, τ=0.5s: v_new = v₀(1 - dt/τ)
        double expectedDecayFactor = 1.0 - 0.1/0.5;  // = 0.8
        assert(abs(p.velocity[0] - expectedDecayFactor) < 1e-10,
            "Expected velocity to decay by first-order approximation v(1 - dt/τ). Expected: %g, Got: %g"
            .format(expectedDecayFactor, p.velocity[0]));
    }
    
    // Test mass independence of decay rate
    {
        double m1 = 1.0;
        double m2 = 2.0;  // Different mass
        double bondStiffness = 1.0;
        double criticalStretch = 0.1;
        double dampingTimeConstant = 0.5;  // Same τ for both
        
        // Create two points with different masses
        auto p1 = new BondBasedPoint!Vector1D(
            Vector1D(0.0),
            m1,
            bondStiffness,
            criticalStretch,
            dampingTimeConstant,
            Vector1D(1.0)  // Same initial velocity
        );
        
        auto p2 = new BondBasedPoint!Vector1D(
            Vector1D(0.0),
            m2,
            bondStiffness,
            criticalStretch,
            dampingTimeConstant,
            Vector1D(1.0)  // Same initial velocity
        );
        
        // Update both points
        p1.updateState([], 0.1);
        p2.updateState([], 0.1);
        
        // Both particles should decay at the same rate despite different masses
        assert(abs(p1.velocity[0] - p2.velocity[0]) < 1e-10,
            "Expected same velocity decay rate regardless of mass. p1 velocity: %g, p2 velocity: %g"
            .format(p1.velocity[0], p2.velocity[0]));
    }

    // Test constant force behavior
    {
        double mass = 1.0;
        double timeStep = 0.1;
        auto constantForce = Vector2D(1.0, 0.0);  // 1N force in x direction
        
        // Create point with constant force
        auto p = new BondBasedPoint!Vector2D(
            Vector2D(0.0, 0.0),
            mass,
            1.0,  // bondStiffness (unused in this test)
            0.1,  // criticalStretch (unused in this test)
            double.infinity,  // No damping
            Vector2D(0.0, 0.0),  // Start at rest
            false,  // Not fixed
            constantForce
        );
        
        // First update - velocity Verlet step
        p.updateState([], timeStep);
        
        // Second update
        p.updateState([], timeStep);
        
        // Under constant force, velocity increases linearly: v = (F/m)t
        // Position follows: x = (F/2m)t²
        double expectedDisplacement = 0.5 * (constantForce[0]/mass) * (2*timeStep)*(2*timeStep);
        assert(abs(p.position[0] - expectedDisplacement) < 1e-10,
            "Position under constant force incorrect. Expected: %g, Got: %g"
            .format(expectedDisplacement, p.position[0]));
        
        // Verify y-position remains unchanged (no force in y direction)
        assert(abs(p.position[1]) < 1e-10,
            "Y position should remain zero under x-direction force. Got: %g"
            .format(p.position[1]));
            
        // Verify force can be changed through property
        p.constantForce = Vector2D(2.0, 0.0);  // Double the force
        assert(abs(p.constantForce[0] - 2.0) < 1e-10,
            "Constant force not updated correctly through property");
    }

    // Test fixed velocity behavior
    {
        double mass = 1.0;
        double bondStiffness = 1000.0;  // Strong bond force
        double criticalStretch = 0.5;
        double timeStep = 0.1;
        
        // Create points with fixed velocity
        auto p1 = new BondBasedPoint!Vector1D(
            Vector1D(0.0),
            mass,
            bondStiffness,
            criticalStretch,
            double.infinity,  // No damping
            Vector1D(1.0),   // Initial velocity: 1.0
            true            // Fixed velocity
        );
        auto p2 = new BondBasedPoint!Vector1D(
            Vector1D(1.0),  // Creates initial force
            mass,
            bondStiffness,
            criticalStretch
        );
        
        // Try to change velocity (should be ignored)
        p1.velocity = Vector1D(2.0);
        assert(abs(p1.velocity[0] - 1.0) < 1e-10,
            "Fixed velocity should not be changeable through setter");
        
        // Update state
        p1.updateState([p2], timeStep);
        
        // Position should update based on fixed velocity
        double expectedPos = 0.0 + 1.0 * timeStep;  // x = x₀ + v*dt
        assert(abs(p1.position[0] - expectedPos) < 1e-10,
            "Position should update based on fixed velocity: Expected %g, Got: %g"
            .format(expectedPos, p1.position[0]));
        
        // Velocity should remain fixed despite forces
        assert(abs(p1.velocity[0] - 1.0) < 1e-10,
            "Velocity should remain fixed despite forces");
        
        // Update again to verify consistent behavior
        p1.updateState([p2], timeStep);
        expectedPos += 1.0 * timeStep;  // Another timestep at v=1.0
        
        assert(abs(p1.position[0] - expectedPos) < 1e-10,
            "Position should continue updating with fixed velocity: Expected %g, Got: %g"
            .format(expectedPos, p1.position[0]));
        assert(abs(p1.velocity[0] - 1.0) < 1e-10,
            "Velocity should remain fixed after multiple updates");
    }

    // Test force ramping behavior
    {
        double mass = 1.0;
        double timeStep = 1e-6;        // 1μs timestep
        double rampDuration = 5e-6;    // 5μs ramp duration
        auto targetForce = Vector2D(0.0, 0.01);  // Target force in y direction
        
        // Create point with ramped force
        auto p = new BondBasedPoint!Vector2D(
            Vector2D(0.0, 0.0),
            mass,
            1.0,            // bondStiffness (unused in this test)
            0.1,            // criticalStretch (unused in this test)
            double.infinity, // No damping
            Vector2D(0.0, 0.0),  // Start at rest
            false,          // Not fixed
            targetForce,    // Target force
            rampDuration   // Ramp duration
        );
        
        // Check initial force (should be zero)
        assert(p.currentForce.magnitude() < 1e-10,
            "Initial force should be zero before first update");
            
        // Update once and check force (should be 20% of target)
        p.updateState([], timeStep);
        double expectedFactor = timeStep / rampDuration;  // 0.2
        assert(abs(p.currentForce[1] - targetForce[1] * expectedFactor) < 1e-10,
            "Force not ramping correctly. Expected %g, Got: %g"
            .format(targetForce[1] * expectedFactor, p.currentForce[1]));
            
        // Update until ramp complete
        for (int i = 0; i < 4; i++) {  // 4 more steps to reach 5μs
            p.updateState([], timeStep);
        }
        
        // Check final force (should equal target)
        assert(abs(p.currentForce[1] - targetForce[1]) < 1e-10,
            "Final force should equal target force. Expected %g, Got: %g"
            .format(targetForce[1], p.currentForce[1]));
            
        // One more update should not increase force beyond target
        p.updateState([], timeStep);
        assert(abs(p.currentForce[1] - targetForce[1]) < 1e-10,
            "Force should remain at target value after ramp completion");
    }
}

module math.vector;

import std.math;

// Generic Vector template for any dimension
struct Vector(size_t N) if (N >= 1 && N <= 3) {
    // Components stored in an array
    double[N] components;
    
    // Constructor with variadic arguments
    this(double[] values...) {
        assert(values.length == N, "Wrong number of components");
        for (size_t i = 0; i < N; i++) {
            components[i] = values[i];
        }
    }
    
    // Default constructor initializes to zero
    static Vector!N zero() {
        Vector!N result;
        result.components[] = 0.0;
        return result;
    }
    
    // Index operator for access
    ref double opIndex(size_t i) {
        assert(i < N, "Index out of bounds");
        return components[i];
    }
    
    // Const index operator for access
    double opIndex(size_t i) const {
        assert(i < N, "Index out of bounds");
        return components[i];
    }
    
    // Vector addition
    Vector!N opBinary(string op)(Vector!N other) const
        if (op == "+") {
        Vector!N result;
        for (size_t i = 0; i < N; i++) {
            result.components[i] = components[i] + other.components[i];
        }
        return result;
    }
    
    // Vector subtraction
    Vector!N opBinary(string op)(Vector!N other) const
        if (op == "-") {
        Vector!N result;
        for (size_t i = 0; i < N; i++) {
            result.components[i] = components[i] - other.components[i];
        }
        return result;
    }
    
    // Scalar multiplication
    Vector!N opBinary(string op)(double scalar) const
        if (op == "*") {
        Vector!N result;
        for (size_t i = 0; i < N; i++) {
            result.components[i] = components[i] * scalar;
        }
        return result;
    }
    
    // Scalar multiplication (scalar on left)
    Vector!N opBinaryRight(string op)(double scalar) const
        if (op == "*") {
        return this * scalar;  // Reuse right multiplication
    }
    
    // Scalar division
    Vector!N opBinary(string op)(double scalar) const
        if (op == "/") {
        Vector!N result;
        for (size_t i = 0; i < N; i++) {
            result.components[i] = components[i] / scalar;
        }
        return result;
    }
    
    // Unary negation
    Vector!N opUnary(string op)() const
        if (op == "-") {
        Vector!N result;
        for (size_t i = 0; i < N; i++) {
            result.components[i] = -components[i];
        }
        return result;
    }
    
    // Dot product
    double dot(Vector!N other) const {
        double sum = 0;
        for (size_t i = 0; i < N; i++) {
            sum += components[i] * other.components[i];
        }
        return sum;
    }
    
    // Magnitude
    double magnitude() const {
        return sqrt(dot(this));
    }
    
    // Unit vector
    Vector!N unit() const {
        double mag = magnitude();
        if (mag > 0) {
            return this * (1.0 / mag);
        } else {
            return Vector!N.zero();
        }
    }
}

// Type aliases for common dimensions
alias Vector1D = Vector!1;
alias Vector2D = Vector!2;
alias Vector3D = Vector!3;

// Unit tests
unittest {
    // Test 1D vector
    auto v1 = Vector1D(3.0);
    assert(v1[0] == 3.0);
    assert(v1.magnitude() == 3.0);
    
    // Test 2D vector
    auto v2a = Vector2D(3.0, 4.0);
    auto v2b = Vector2D(1.0, 2.0);
    assert(v2a.magnitude() == 5.0);
    assert((v2a + v2b)[0] == 4.0);
    assert((v2a + v2b)[1] == 6.0);
    
    // Test 3D vector
    auto v3 = Vector3D(1.0, 2.0, 2.0);
    assert(v3.magnitude() == 3.0);
    auto unit = v3.unit();
    assert(abs(unit.magnitude() - 1.0) < 1e-10);
    
    // Test scalar multiplication
    auto v = Vector2D(2.0, 3.0);
    auto scaled = v * 2.0;
    assert(scaled[0] == 4.0);
    assert(scaled[1] == 6.0);
    
    // Test dot product
    auto v4 = Vector2D(1.0, 2.0);
    auto v5 = Vector2D(3.0, 4.0);
    assert(v4.dot(v5) == 11.0);
    
    // Test scalar division
    auto v6 = Vector2D(4.0, 6.0);
    auto divided = v6 / 2.0;
    assert(divided[0] == 2.0);
    assert(divided[1] == 3.0);
    
    // Test negation
    auto v7 = Vector2D(1.0, -2.0);
    auto negated = -v7;
    assert(negated[0] == -1.0);
    assert(negated[1] == 2.0);
}

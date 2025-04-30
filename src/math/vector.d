module math.vector;

import std.math;
import core.simd;
import core.cpuid;

// CPU feature detection
static bool hasSSE2;
static bool hasAVX;

static this() {
    hasSSE2 = core.cpuid.sse2;
    hasAVX = core.cpuid.avx;
    import std.stdio;
    writefln("SIMD Support Detected:\n  Runtime - SSE2: %s, AVX: %s\n  Compile-time - SSE2: %s, AVX: %s", 
        hasSSE2, hasAVX,
        is(double2) ? "yes" : "no",
        is(double4) ? "yes" : "no");
}

// Generic Vector template for any dimension
struct Vector(size_t N) if (N >= 1 && N <= 3) {
    static if (N == 2) {
        static if (is(double2)) {  // Check if SIMD type is available
            union {
                double[2] components;
                double2 simdComponents;
            }
        } else {
            double[2] components;
        }
    } else static if (N == 3) {
        static if (is(double4)) {  // AVX double vector
            union {
                double[3] components;
                double4 simdComponents;  // We'll only use first 3 components
            }
        } else {
            double[3] components;
        }
    } else {
        double[N] components;  // 1D case remains scalar
    }
    
    // Constructor with variadic arguments
    this(double[] values...) {
        assert(values.length == N, "Wrong number of components");
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                components[0] = values[0];
                components[1] = values[1];
                return;
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                components[0] = values[0];
                components[1] = values[1];
                components[2] = values[2];
                return;
            }
        }
        foreach (i; 0..N) {
            components[i] = values[i];
        }
    }
    
    // Default constructor initializes to zero
    static Vector!N zero() {
        Vector!N result;
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                result.simdComponents = 0;
                return result;
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                result.simdComponents = 0;
                return result;
            }
        }
        result.components[] = 0;
        return result;
    }
    
    // Index operator for access
    ref double opIndex(size_t i) {
        assert(i < N, "Index out of bounds");
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                return *(&simdComponents[0] + i);
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                return *(&simdComponents[0] + i);
            }
        }
        return components[i];
    }
    
    // Const index operator for access
    double opIndex(size_t i) const {
        assert(i < N, "Index out of bounds");
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                return simdComponents[i];
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                return simdComponents[i];
            }
        }
        return components[i];
    }
    
    // Vector addition with SIMD support
    Vector!N opBinary(string op)(Vector!N other) const
        if (op == "+") {
        Vector!N result;
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                result.simdComponents = simdComponents + other.simdComponents;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = components[i] + other.components[i];
                }
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                result.simdComponents = simdComponents + other.simdComponents;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = components[i] + other.components[i];
                }
            }
        } else {
            foreach (i; 0..N) {
                result.components[i] = components[i] + other.components[i];
            }
        }
        return result;
    }
    
    // Vector subtraction with SIMD support
    Vector!N opBinary(string op)(Vector!N other) const
        if (op == "-") {
        Vector!N result;
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                result.simdComponents = simdComponents - other.simdComponents;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = components[i] - other.components[i];
                }
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                result.simdComponents = simdComponents - other.simdComponents;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = components[i] - other.components[i];
                }
            }
        } else {
            foreach (i; 0..N) {
                result.components[i] = components[i] - other.components[i];
            }
        }
        return result;
    }
    
    // Scalar multiplication with SIMD support
    Vector!N opBinary(string op)(double scalar) const
        if (op == "*") {
        Vector!N result;
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                result.simdComponents = simdComponents * scalar;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = components[i] * scalar;
                }
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                result.simdComponents = simdComponents * scalar;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = components[i] * scalar;
                }
            }
        } else {
            foreach (i; 0..N) {
                result.components[i] = components[i] * scalar;
            }
        }
        return result;
    }
    
    // Scalar multiplication (scalar on left)
    Vector!N opBinaryRight(string op)(double scalar) const
        if (op == "*") {
        return this * scalar;  // Reuse right multiplication
    }
    
    // Scalar division with SIMD support
    Vector!N opBinary(string op)(double scalar) const
        if (op == "/") {
        Vector!N result;
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                result.simdComponents = simdComponents / scalar;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = components[i] / scalar;
                }
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                result.simdComponents = simdComponents / scalar;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = components[i] / scalar;
                }
            }
        } else {
            foreach (i; 0..N) {
                result.components[i] = components[i] / scalar;
            }
        }
        return result;
    }
    
    // Unary negation with SIMD support
    Vector!N opUnary(string op)() const
        if (op == "-") {
        Vector!N result;
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                result.simdComponents = -simdComponents;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = -components[i];
                }
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                result.simdComponents = -simdComponents;
            } else {
                foreach (i; 0..N) {
                    result.components[i] = -components[i];
                }
            }
        } else {
            foreach (i; 0..N) {
                result.components[i] = -components[i];
            }
        }
        return result;
    }
    
    // Dot product with SIMD support
    double dot(Vector!N other) const {
        static if (N == 2 && is(double2)) {
            if (hasSSE2) {
                auto prod = simdComponents * other.simdComponents;
                return prod[0] + prod[1];
            } else {
                double sum = 0;
                foreach (i; 0..N) {
                    sum += components[i] * other.components[i];
                }
                return sum;
            }
        } else static if (N == 3 && is(double4)) {
            if (hasAVX) {
                auto prod = simdComponents * other.simdComponents;
                return prod[0] + prod[1] + prod[2];
            } else {
                double sum = 0;
                foreach (i; 0..N) {
                    sum += components[i] * other.components[i];
                }
                return sum;
            }
        } else {
            double sum = 0;
            foreach (i; 0..N) {
                sum += components[i] * other.components[i];
            }
            return sum;
        }
    }
    
    // Magnitude squared with SIMD support
    double magnitudeSquared() const {
        return dot(this);  // Reuse existing dot product implementation
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

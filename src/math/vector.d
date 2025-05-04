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

// Generic Vector template with compile-time or runtime dimension
struct Vector(size_t N = 0) {
    // Static dimension for compile-time sized vectors
    static if (N > 0) {
        enum dimension = N;
        
        // Select storage type at compile time
        static if (N == 2 && is(double2))
            private double2 _components;
        else static if (N == 3 && is(double4))
            private double4 _components;
        else
            private double[N] _components;
            
    } else {
        private double[] _components;
        private size_t _dimension;
        @property size_t dimension() const { return _dimension; }
    }
    
    // Constructor for compile-time dimension
    static if (N > 0) {
        this(double[] values...) {
            assert(values.length == N, "Wrong number of components");
            static if (N == 2 && is(double2)) {
                _components[0] = values[0];
                _components[1] = values[1];
            } else static if (N == 3 && is(double4)) {
                _components[0] = values[0];
                _components[1] = values[1];
                _components[2] = values[2];
            } else {
                _components[0..N] = values[0..N];
            }
        }
    }
    // Constructors for runtime dimension
    else {
        this(size_t dim) {
            _dimension = dim;
            _components = new double[dim];
        }
        
        this(double[] values) {
            _dimension = values.length;
            _components = values.dup;
        }
    }
    
    // Create zero vector
    static Vector!N zero() {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = 0;
            } else static if (N == 3 && is(double4)) {
                result._components = 0;
            } else {
                result._components[] = 0;
            }
            return result;
        } else {
            return Vector!N(0);
        }
    }
    
    // Create zero vector with runtime dimension
    static Vector!N zero(size_t dim) {
        static if (N == 0) {
            auto result = Vector!N(dim);
            result._components[] = 0;
            return result;
        } else {
            assert(dim == N, "Dimension mismatch");
            return zero();
        }
    }
    
    // Index operator
    ref double opIndex(size_t i) {
        static if (N > 0) {
            assert(i < N, "Index out of bounds");
        } else {
            assert(i < _dimension, "Index out of bounds");
        }
        return _components[i];
    }
    
    // Const index operator
    double opIndex(size_t i) const {
        static if (N > 0) {
            assert(i < N, "Index out of bounds");
        } else {
            assert(i < _dimension, "Index out of bounds");
        }
        return _components[i];
    }
    
    // Vector addition
    Vector!N opBinary(string op)(Vector!N other) const
        if (op == "+") {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = _components + other._components;
            } else static if (N == 3 && is(double4)) {
                result._components = _components + other._components;
            } else {
                result._components[] = _components[] + other._components[];
            }
            return result;
        } else {
            assert(_dimension == other._dimension, "Dimension mismatch");
            auto result = Vector!N(_dimension);
            result._components[] = _components[] + other._components[];
            return result;
        }
    }
    
    // Vector subtraction
    Vector!N opBinary(string op)(Vector!N other) const
        if (op == "-") {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = _components - other._components;
            } else static if (N == 3 && is(double4)) {
                result._components = _components - other._components;
            } else {
                result._components[] = _components[] - other._components[];
            }
            return result;
        } else {
            assert(_dimension == other._dimension, "Dimension mismatch");
            auto result = Vector!N(_dimension);
            result._components[] = _components[] - other._components[];
            return result;
        }
    }
    
    // Scalar multiplication
    Vector!N opBinary(string op)(double scalar) const
        if (op == "*") {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = _components * scalar;
            } else static if (N == 3 && is(double4)) {
                result._components = _components * scalar;
            } else {
                result._components[] = _components[] * scalar;
            }
            return result;
        } else {
            auto result = Vector!N(_dimension);
            result._components[] = _components[] * scalar;
            return result;
        }
    }
    
    // Scalar multiplication (scalar on left)
    Vector!N opBinaryRight(string op)(double scalar) const
        if (op == "*") {
        return this * scalar;
    }
    
    // Scalar division
    Vector!N opBinary(string op)(double scalar) const
        if (op == "/") {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = _components / scalar;
            } else static if (N == 3 && is(double4)) {
                result._components = _components / scalar;
            } else {
                result._components[] = _components[] / scalar;
            }
            return result;
        } else {
            auto result = Vector!N(_dimension);
            result._components[] = _components[] / scalar;
            return result;
        }
    }
    
    // Unary negation
    Vector!N opUnary(string op)() const
        if (op == "-") {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = -_components;
            } else static if (N == 3 && is(double4)) {
                result._components = -_components;
            } else {
                result._components[] = -_components[];
            }
            return result;
        } else {
            auto result = Vector!N(_dimension);
            result._components[] = -_components[];
            return result;
        }
    }
    
    // Dot product
    double dot(const(Vector!N) other) const {
        static if (N > 0) {
            static if (N == 2 && is(double2)) {
                auto prod = _components * other._components;
                return prod[0] + prod[1];
            } else static if (N == 3 && is(double4)) {
                auto prod = _components * other._components;
                return prod[0] + prod[1] + prod[2];
            } else {
                double sum = 0;
                foreach (i; 0..N) {
                    sum += _components[i] * other._components[i];
                }
                return sum;
            }
        } else {
            assert(_dimension == other._dimension, "Dimension mismatch");
            double sum = 0;
            foreach (i; 0.._dimension) {
                sum += _components[i] * other._components[i];
            }
            return sum;
        }
    }
    
    // Magnitude squared
    double magnitudeSquared() const {
        return dot(this);
    }
    
    // Magnitude
    double magnitude() const {
        return sqrt(magnitudeSquared());
    }
    
    // Unit vector
    Vector!N unit() const {
        double mag = magnitude();
        if (mag > 0) {
            return this * (1.0 / mag);
        } else {
            static if (N > 0) {
                return Vector!N.zero();
            } else {
                return Vector!N.zero(_dimension);
            }
        }
    }
    
    // Equality comparison
    bool opEquals(const Vector!N other) const {
        static if (N > 0) {
            static if (N == 2 && is(double2)) {
                return _components[0] == other._components[0] && 
                       _components[1] == other._components[1];
            } else static if (N == 3 && is(double4)) {
                return _components[0] == other._components[0] && 
                       _components[1] == other._components[1] && 
                       _components[2] == other._components[2];
            } else {
                foreach (i; 0..N) {
                    if (_components[i] != other._components[i]) return false;
                }
                return true;
            }
        } else {
            if (_dimension != other._dimension) return false;
            foreach (i; 0.._dimension) {
                if (_components[i] != other._components[i]) return false;
            }
            return true;
        }
    }
    
    // Safe iteration over components
    int opApply(int delegate(size_t i, ref double value) dg) {
        static if (N > 0) {
            foreach (i; 0..N) {
                if (auto result = dg(i, _components[i]))
                    return result;
            }
        } else {
            foreach (i; 0.._dimension) {
                if (auto result = dg(i, _components[i]))
                    return result;
            }
        }
        return 0;
    }
    
    // Const iteration over components
    int opApply(int delegate(size_t i, const ref double value) dg) const {
        static if (N > 0) {
            foreach (i; 0..N) {
                if (auto result = dg(i, _components[i]))
                    return result;
            }
        } else {
            foreach (i; 0.._dimension) {
                if (auto result = dg(i, _components[i]))
                    return result;
            }
        }
        return 0;
    }
}

// Type aliases for common dimensions
alias Vector1D = Vector!1;
alias Vector2D = Vector!2;
alias Vector3D = Vector!3;
alias DynamicVector = Vector!0;

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
    
    // Test dynamic vector
    auto dv1 = DynamicVector([1.0, 2.0, 3.0]);
    auto dv2 = DynamicVector([4.0, 5.0, 6.0]);
    assert(dv1.dimension == 3);
    assert(dv1.dot(dv2) == 32.0);
    
    // Test scalar operations
    auto scaled = dv1 * 2.0;
    assert(scaled[0] == 2.0);
    assert(scaled[1] == 4.0);
    assert(scaled[2] == 6.0);
    
    // Test vector operations
    auto sum = dv1 + dv2;
    assert(sum[0] == 5.0);
    assert(sum[1] == 7.0);
    assert(sum[2] == 9.0);
    
    // Test iteration
    double sumComponents = 0;
    foreach (i, value; dv1) {
        sumComponents += value;
    }
    assert(sumComponents == 6.0);
}

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
    Vector!N opBinary(string op)(const Vector!N other) const
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
    Vector!N opBinary(string op)(const Vector!N other) const
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

    // Create a copy of this vector
    Vector!N dup() const {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = _components;
            } else static if (N == 3 && is(double4)) {
                result._components = _components;
            } else {
                result._components[] = _components[];
            }
            return result;
        } else {
            auto result = Vector!N(_dimension);
            result._components[] = _components[];
            return result;
        }
    }

    // Hadamard (element-wise) multiplication
    Vector!N hadamard(Vector!N other) const {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = _components * other._components;
            } else static if (N == 3 && is(double4)) {
                result._components = _components * other._components;
            } else {
                foreach (i; 0..N) {
                    result._components[i] = _components[i] * other._components[i];
                }
            }
            return result;
        } else {
            assert(_dimension == other._dimension, "Dimension mismatch");
            auto result = Vector!N(_dimension);
            foreach (i; 0.._dimension) {
                result._components[i] = _components[i] * other._components[i];
            }
            return result;
        }
    }

    // Serialize vector to bytes
    ubyte[] toBytes() const {
        ubyte[] buffer = [69]; // Canary value

        static if (N > 0) {
            // For compile-time dimension, components are fixed size
            static if (N == 2 && is(double2)) {
                buffer ~= (cast(ubyte*)&_components[0])[0..double.sizeof * 2];
            } else static if (N == 3 && is(double4)) {
                buffer ~= (cast(ubyte*)&_components[0])[0..double.sizeof * 3];
            } else {
                buffer ~= (cast(ubyte*)_components.ptr)[0..double.sizeof * N];
            }
        } else {
            // For runtime dimension, include size and components
            auto size = _dimension;
            buffer ~= cast(ubyte[])(&size)[0..1];
            buffer ~= (cast(ubyte*)_components.ptr)[0..double.sizeof * _dimension];
        }
        
        return buffer;
    }

    // Deserialize vector from bytes
    // Deserialize vector from bytes into target vector
    static size_t fromBytes(ubyte[] data, ref Vector!N target) {
        assert(data.length > 0, "Data too short for canary value");
        assert(data[0] == 69, "Invalid vector data - incorrect canary value");
        data = data[1..$]; // Skip past canary

        static if (N > 0) {
            // For fixed dimension, read N components and use varargs constructor
            auto values = (cast(double*)data.ptr)[0..N];
            target = Vector!N(values[]);
            return 1 + double.sizeof * N; // Add 1 for canary byte
        } else {
            // For runtime dimension, read size and components
            size_t size = *cast(size_t*)data.ptr;
            size_t offset = size_t.sizeof;
            size_t bytes_read = size * double.sizeof;

            assert(data.length >= bytes_read, "Data too short");

            auto values = (cast(double*)(data.ptr + offset))[0..size];
            target = Vector!N(values.dup);
            return 1 + offset + double.sizeof * size; // Add 1 for canary byte
        }
    }

    // Deserialize vector from bytes
    static Vector!N fromBytes(ubyte[] data) {
        Vector!N result;
        fromBytes(data, result);
        return result;
    }

    // Hadamard (element-wise) division
    Vector!N hadamardDiv(Vector!N other) const {
        static if (N > 0) {
            Vector!N result;
            static if (N == 2 && is(double2)) {
                result._components = _components / other._components;
            } else static if (N == 3 && is(double4)) {
                result._components = _components / other._components;
            } else {
                foreach (i; 0..N) {
                    result._components[i] = _components[i] / other._components[i];
                }
            }
            return result;
        } else {
            assert(_dimension == other._dimension, "Dimension mismatch");
            auto result = Vector!N(_dimension);
            foreach (i; 0.._dimension) {
                result._components[i] = _components[i] / other._components[i];
            }
            return result;
        }
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

    // Test dup
    auto v2_orig = Vector2D(1.0, 2.0);
    auto v2_copy = v2_orig.dup();
    assert(v2_orig == v2_copy);
    v2_copy[0] = 3.0;
    assert(v2_orig[0] == 1.0);  // Original unchanged

    auto dv_orig = DynamicVector([1.0, 2.0, 3.0]);
    auto dv_copy = dv_orig.dup();
    assert(dv_orig == dv_copy);
    dv_copy[1] = 5.0;
    assert(dv_orig[1] == 2.0);  // Original unchanged

    // Test serialization/deserialization
    
    // Test canary value validation
    {
        // Test invalid canary value
        auto v1_bad = Vector1D(3.14);
        auto v1_bytes_bad = v1_bad.toBytes();
        v1_bytes_bad[0] = 42; // Corrupt canary value
        bool threw = false;
        try {
            auto v1_restored = Vector1D.fromBytes(v1_bytes_bad);
        } catch (Error e) {
            threw = true;
            assert(e.msg == "Invalid vector data - incorrect canary value");
        }
        assert(threw, "Should throw on invalid canary");
    }

    // Test empty data
    {
        bool threw = false;
        try {
            auto v_empty = Vector1D.fromBytes([]);
        } catch (Error e) {
            threw = true;
            assert(e.msg == "Data too short for canary value");
        }
        assert(threw, "Should throw on empty data");
    }
    
    // Fixed dimension vectors
    auto v1_ser = Vector1D(3.14);
    auto v1_bytes = v1_ser.toBytes();
    auto v1_restored = Vector1D.fromBytes(v1_bytes);
    assert(v1_ser == v1_restored);
    
    auto v2_ser = Vector2D(1.23, 4.56);
    auto v2_bytes = v2_ser.toBytes();
    auto v2_restored = Vector2D.fromBytes(v2_bytes);
    assert(v2_ser == v2_restored);
    
    auto v3_ser = Vector3D(1.1, 2.2, 3.3);
    auto v3_bytes = v3_ser.toBytes();
    auto v3_restored = Vector3D.fromBytes(v3_bytes);
    assert(v3_ser == v3_restored);
    
    // Dynamic vector
    auto dv_ser = DynamicVector([1.1, 2.2, 3.3, 4.4]);
    auto dv_bytes = dv_ser.toBytes();
    auto dv_restored = DynamicVector.fromBytes(dv_bytes);
    assert(dv_ser == dv_restored);
    assert(dv_ser.dimension == dv_restored.dimension);
    
    // Test reference fromBytes
    Vector1D v1_ref;
    size_t v1_bytes_read = Vector1D.fromBytes(v1_bytes, v1_ref);
    assert(v1_bytes_read == 1 + double.sizeof); // Add 1 for canary byte
    assert(v1_ref == v1_ser);
    
    Vector2D v2_ref;
    size_t v2_bytes_read = Vector2D.fromBytes(v2_bytes, v2_ref);
    assert(v2_bytes_read == 1 + double.sizeof * 2); // Add 1 for canary byte
    assert(v2_ref == v2_ser);
    
    Vector3D v3_ref;
    size_t v3_bytes_read = Vector3D.fromBytes(v3_bytes, v3_ref);
    assert(v3_bytes_read == 1 + double.sizeof * 3); // Add 1 for canary byte
    assert(v3_ref == v3_ser);
    
    DynamicVector dv_ref;
    size_t dv_bytes_read = DynamicVector.fromBytes(dv_bytes, dv_ref);
    assert(dv_bytes_read == 1 + size_t.sizeof + double.sizeof * dv_ser.dimension); // Add 1 for canary byte
    assert(dv_ref == dv_ser);
    assert(dv_ref.dimension == dv_ser.dimension);
}

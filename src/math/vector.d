module math.vector;

import std.math;
import std.complex;
import std.format;
import std.traits;
import core.simd;
import core.cpuid;
import std.conv;
import std.bitmanip;

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

template isComplex(T) {
    static if (is(T == Complex!U, U))
        enum isComplex = true;
    else
        enum isComplex = false;
}

template isDouble2(T) {
    static if (is(double2) && is(T == double2))
        enum isDouble2 = true;
    else
        enum isDouble2 = false;
}

template isDouble4(T) {
    static if (is(double4) && is(T == double4))
        enum isDouble4 = true;
    else
        enum isDouble4 = false;
}

template isFloat2(T) {
    static if (is(float2) && is(T == float2))
        enum isFloat2 = true;
    else
        enum isFloat2 = false;
}

template isFloat4(T) {
    static if (is(float4) && is(T == float4))
        enum isFloat4 = true;
    else
        enum isFloat4 = false;
}

template isSimd2(T) {
    static if (isDouble2!(T) || isFloat2!(T)) {
        enum isSimd2 = true;
    } else {
        enum isSimd2 = false;
    }
}

template isSimd4(T) {
    static if (isDouble4!(T) || isFloat4!(T)) {
        enum isSimd4 = true;
    } else {
        enum isSimd4 = false;
    }
}

template isSimd(T) {
    static if (isDouble2!(T) || isDouble4!(T) || isFloat2!(T) || isFloat4!(T)) {
        enum isSimd = true;
    } else {
        enum isSimd = false;
    }
}

// Vector type checking
template isVector(V) {
    enum isVector = is(V == Vector!(T, N), T, size_t N);
}

// Generic Vector template with compile-time or runtime dimension
struct Vector(T, size_t N = 0)
if (isNumeric!T || isComplex!T) {
    // Static dimension for compile-time sized vectors
    static if (N > 0) {
        
        // Select storage type at compile time
        static if (is(T == double) && N == 2 && is(double2))
            private double2 _components;
        else static if (is(T == double) && (N == 3 || N == 4) && is(double4))
            private double4 _components;
        else static if (is(T == float) && N == 2 && is(float2))
            private float2 _components;
        else static if (is(T == float) && (N == 3 || N == 4) && is(float4))
            private float4 _components;
        else
            private T[N] _components;
            
    } else {
        private T[] _components;
    }
    
    // Constructor for compile-time dimension
    static if (N > 0) {
        this(T[] values...) {
            assert(values.length == N, "Wrong number of components");
            _components[0..N] = values[0..N];
        }
    }
    else {
        
        this(T[] values) {
            _components = values.dup;
        }
    }

    this(size_t dim) {
        static if (N == 0) {
            _components = new T[dim];
        }
        else {
            assert(dim == N, "Dimension mismatch for compile-time vector");
        }
    }
    
    // Create zero vector
    static Vector!(T, N) zero(size_t dim = N) {
        static if (N > 0) {
            Vector!(T, N) result;
            assert(dim == N, "Dimension mismatch for compile-time vector");
        }
        else {
            Vector!(T, N) result = Vector!(T,N)(dim);
        }

        static if (isSimd!(typeof(_components))) {
            result._components = 0;
        }
        else static if (isComplex!T) {
            result._components[] = T(0, 0);
        } else {
            result._components[] = 0;
        }
        return result;
    }
    
    // String representation
    string toString() const {
        // Open bracket
        auto result = "[";
        
        // Add components with appropriate formatting
        foreach(i, component; _components) {
            if (i > 0) result ~= ", ";
            
            static if (isComplex!T)
                result ~= format("%s", component);  // Complex numbers 
            else
                result ~= format("%g", component);  // Regular numbers use %g for clean formatting
        }
        
        // Close bracket
        result ~= "]";
        return result;
    }

    // Index operator
    ref T opIndex(size_t i) {
        assert(i < length, "Index out of bounds");
        return _components[i];
    }
    
    // Const index operator
    T opIndex(size_t i) const {
        assert(i < length, "Index out of bounds");
        return _components[i];
    }
    
    // Vector element-wise operations
    Vector!(T, N) opBinary(string op)(const Vector!(T, N) other) const
        if (op == "+" || op == "-" || op == "/" || op == "*") {
        Vector!(T, N) result = Vector!(T, N)(length);
        static if (isSimd!(typeof(_components))) {
            result._components = mixin("_components " ~ op ~ " other._components");
        } else {
            assert(length == other.length, "Dimension mismatch");
            result._components[] = mixin("_components[] " ~ op ~ " other._components[]");
        }
        return result;
    }
    
    // vector-scalar operations
    Vector!(T, N) opBinary(string op)(T scalar) const
        if (op == "*" || op == "/") {
        Vector!(T, N) result = Vector!(T, N)(length);
        static if (isSimd!(typeof(_components))) {
            result._components = mixin("_components " ~ op ~ " scalar");
        } else {
            result._components[] = mixin("_components[] " ~ op ~ " scalar");
        }
        return result;
    }
    
    // scalar-vector operations
    Vector!(T, N) opBinaryRight(string op)(T scalar) const
        if (op == "*" ) {
        return mixin("this " ~ op ~ " scalar");
    }

    // In-place vector-vector operations
    void opOpAssign(string op)(const Vector!(T, N) other)
        if (op == "+" || op == "-" || op == "/" || op == "*") {
        static if (isSimd!(typeof(_components))) {
            mixin("_components " ~ op ~ "= other._components;");
        } else {
            assert(length == other.length, "Dimension mismatch");
            mixin("_components[] " ~ op ~ "= other._components[];");
        }
    }

    // In-place vector-scalar operations  
    void opOpAssign(string op)(T scalar)
        if (op == "*" || op == "/") {
        static if (isSimd!(typeof(_components))) {
            mixin("_components " ~ op ~ "= scalar;");
        } else {
            mixin("_components[] " ~ op ~ "= scalar;");
        }
    }
    
    // Unary negation
    Vector!(T, N) opUnary(string op)() const
        if (op == "-") {
        Vector!(T, N) result = Vector!(T, N)(length);
        static if (isSimd!(typeof(_components))) {
            result._components = -_components;
        } else {
            result._components[] = -_components[];
        }
        return result;
    }
    
    // Dot product
    auto dot(const(Vector!(T, N)) other) const {
        static if (isComplex!T) {
            static if (N > 0) {
                T sum = T(0);
                foreach (i; 0..N) {
                    sum += conj(this[i]) * other[i];
                }
                return sum;
            } else {
                assert(length == other.length, "Dimension mismatch");
                T sum = T(0);
                foreach (i; 0..length) {
                    sum += conj(this[i]) * other[i];
                }
                return sum;
            }
        } else {
            T sum = 0;
            auto prod = this.hadamard(other);
            foreach (i; 0..length) {
                sum += prod[i];
            }
            return sum;
        }
    }
    // Magnitude squared
    auto magnitudeSquared() const {
        static if (isComplex!T) {
            auto d = dot(this);
            return d.re;
        } else {
            return dot(this);
        }
    }
    
    // Magnitude
    auto magnitude() const {
        static if (isComplex!T) {
            return sqrt(magnitudeSquared());
        } else static if (isFloatingPoint!T) {
            return sqrt(magnitudeSquared());
        } else {
            return cast(T)sqrt(cast(real)magnitudeSquared());
        }
    }
    
    // Unit vector
    Vector!(T, N) unit() const {
        auto mag = magnitude();
        if (mag > 0) {
            return this * (T(1) / T(mag));
        }
        else {
            return Vector!(T, N).zero(length);
        }
        
    }
    
    // Equality comparison
    bool opEquals(const Vector!(T, N) other) const {
        if (length != other.length) return false;
        foreach (i; 0..length) {
            if (_components[i] != other._components[i]) return false;
        }
        return true;
    }
    
    // Safe iteration over components
    int opApply(int delegate(size_t i, ref T value) dg) {
        foreach (i; 0..length) {
            if (auto result = dg(i, _components[i]))
                return result;
        }
        return 0;
    }
    
    // Const iteration over components
    int opApply(int delegate(size_t i, const ref T value) dg) const {
        foreach (i; 0..length) {
            if (auto result = dg(i, _components[i]))
                return result;
        }
        return 0;
    }

    // Create a copy of this vector
    Vector!(T, N) dup() const {
        Vector!(T, N) result = Vector!(T, N)(length);
        static if (isSimd!(typeof(_components))) {
            result._components = _components;
        } else {
            result._components[] = _components[];
        }
        return result;
    }

    // Hadamard (element-wise) multiplication
    Vector!(T, N) hadamard(const Vector!(T, N) other) const {
        return this * other;
    }
    
    static if (N > 0) {
        // Compile-time dimension, use fixed size
        static size_t length() {
            return N;
        }
    } else {
        // Dynamic dimension, use array length
        size_t length() const {
            return _components.length;
        }
    }

    // Serialize vector to bytes
    ubyte[] toBytes() const {
        ubyte[] buffer = [69]; // Canary value
        buffer ~= nativeToBigEndian(cast(long)this.length); // Append dimension size

        // For compile-time dimension, components are fixed size
        size_t index = 0;
        static if (isComplex!T) {
            ubyte[] componentBuffer = new ubyte[T.sizeof * length];
            foreach (i; 0..length) {
                write!(typeof(T.init.re))(componentBuffer, _components[i].re, &index);
                write!(typeof(T.init.im))(componentBuffer, _components[i].im, &index);
            }
            buffer ~= componentBuffer;
        } else {
            ubyte[] componentBuffer = new ubyte[T.sizeof * length];
            foreach (i; 0..length) {
                write!T(componentBuffer, _components[i], &index);
            }
            buffer ~= componentBuffer;
        }

        return buffer;
    }

    // Deserialize vector from bytes into target vector
    static size_t fromBytes(ubyte[] data, ref Vector!(T, N) target) {
        assert(data.length > 0, "Data too short for canary value");
        assert(data[0] == 69, "Invalid vector data - incorrect canary value");
        data = data[1..$]; // Skip past canary
        
        assert(data.length >= long.sizeof, "Data too short for dimension size");
        const dim = data.read!long();
        assert(dim >= 0, "Invalid vector dimension size");
        
        static if (isComplex!T) {
            const componentSize = T.sizeof;
            const bytesNeeded = componentSize * dim;
            assert(data.length >= bytesNeeded, "Data too short for complex vector components");

            Vector!(T, N) values = Vector!(T, N)(dim);
            foreach (i; 0..dim) {
                auto re = data.read!(typeof(T.init.re))();
                auto im = data.read!(typeof(T.init.im))();
                values[i] = T(re, im);
            }
            target = values;
            return 1 + long.sizeof + bytesNeeded;
        } else {
            const bytesNeeded = T.sizeof * dim;
            assert(data.length >= bytesNeeded, "Data too short for vector components");

            Vector!(T, N) values = Vector!(T, N)(dim);
            foreach (i; 0..dim) {
                values[i] = data.read!T();
            }
            target = values;
            return 1 + long.sizeof + bytesNeeded;
        }
    }

    // Deserialize vector from bytes
    static Vector!(T, N) fromBytes(ubyte[] data) {
        Vector!(T, N) result;
        fromBytes(data, result);
        return result;
    }

    // Hadamard (element-wise) division
    Vector!(T, N) hadamardDiv(const Vector!(T, N) other) const {
        return this / other;
    }
}

// Type aliases for common dimensions
alias Vector1D(T = double) = Vector!(T, 1);
alias Vector2D(T = double) = Vector!(T, 2);
alias Vector3D(T = double) = Vector!(T, 3);
alias DynamicVector(T = double) = Vector!(T, 0);

// Unit tests
unittest {
    // Test isVector template
    static assert(isVector!(Vector!(double, 1)));
    static assert(isVector!(Vector!(double, 2)));
    static assert(isVector!(Vector!(double, 3)));
    static assert(isVector!(Vector!(double, 0)));  // Dynamic vector
    static assert(isVector!(Vector!(float, 2)));   // Different numeric type
    static assert(isVector!(Vector!(int, 3)));     // Integer type
    static assert(isVector!(Vector!(Complex!double, 2))); // Complex type
    static assert(!isVector!int);
    static assert(!isVector!double);
    static assert(!isVector!(int[]));
    static assert(!isVector!(double[]));

    import std.complex;
    import std.math.operations : isClose;
    import std.math;

    // Test basic double vectors (ensure backwards compatibility)
    {
        // Test 1D vector
        auto v1 = Vector1D!double(3.0);
        assert(v1[0] == 3.0, "1D vector access failed, expected: 3.0, got: " ~ to!string(v1[0]));
        assert(v1.magnitude() == 3.0, "1D vector magnitude failed, expected: 3.0, got: " ~ to!string(
                v1.magnitude()));

        // Test 2D vector
        auto v2a = Vector2D!double(3.0, 4.0);
        auto v2b = Vector2D!double(1.0, 2.0);
        assert(v2a.magnitude() == 5.0, "2D vector magnitude failed, expected: 5.0, got: " ~ to!string(
                v2a.magnitude()));
        assert((v2a + v2b)[0] == 4.0, "2D vector addition x-component failed, expected: 4.0, got: " ~ to!string(
                (v2a + v2b)[0]));
        assert((v2a + v2b)[1] == 6.0, "2D vector addition y-component failed, expected: 6.0, got: " ~ to!string(
                (v2a + v2b)[1]));

        // Test 3D vector
        auto v3 = Vector3D!double(1.0, 2.0, 2.0);
        assert(v3.magnitude() == 3.0, "3D vector magnitude failed, expected: 3.0, got: " ~ to!string(
                v3.magnitude()));
        auto unit = v3.unit();
        assert(abs(unit.magnitude() - 1.0) < 1e-10, "Unit vector magnitude deviation: " ~ to!string(
                abs(unit.magnitude() - 1.0)));

        // Test dynamic vector
        auto dv1 = DynamicVector!double([1.0, 2.0, 3.0]);
        auto dv2 = DynamicVector!double([4.0, 5.0, 6.0]);
        assert(dv1.length == 3, "Dynamic vector dimension failed, expected: 3, got: " ~ to!string(
                dv1.length));
        assert(dv1.dot(dv2) == 32.0, "Dynamic vector dot product failed, expected: 32.0, got: " ~ to!string(
                dv1.dot(dv2)));

        // Test scalar operations
        auto basicScaled = dv1 * 2.0;
        assert(basicScaled[0] == 2.0, "Vector scalar multiplication failed for index 0, expected: 2.0, got: " ~ to!string(
                basicScaled[0]));
        assert(basicScaled[1] == 4.0, "Vector scalar multiplication failed for index 1, expected: 4.0, got: " ~ to!string(
                basicScaled[1]));
        assert(basicScaled[2] == 6.0, "Vector scalar multiplication failed for index 2, expected: 6.0, got: " ~ to!string(
                basicScaled[2]));

        // Test vector operations
        auto basicSum = dv1 + dv2;
        assert(basicSum[0] == 5.0, "Vector addition failed for index 0, expected: 5.0, got: " ~ to!string(
                basicSum[0]));
        assert(basicSum[1] == 7.0, "Vector addition failed for index 1, expected: 7.0, got: " ~ to!string(
                basicSum[1]));
        assert(basicSum[2] == 9.0, "Vector addition failed for index 2, expected: 9.0, got: " ~ to!string(
                basicSum[2]));

        // Test iteration
        double sumComponents = 0;
        foreach (i, value; dv1)
        {
            sumComponents += value;
        }
        assert(sumComponents == 6.0, "Vector component sum failed, expected: 6.0, got: " ~ to!string(
                sumComponents));

        // Test dup
        auto v2_orig = Vector2D!double(1.0, 2.0);
        auto v2_copy = v2_orig.dup();
        assert(v2_orig == v2_copy);
        v2_copy[0] = 3.0;
        assert(v2_orig[0] == 1.0); // Original unchanged

        auto dv_orig = DynamicVector!double([1.0, 2.0, 3.0]);
        auto dv_copy = dv_orig.dup();
        assert(dv_orig == dv_copy);
        dv_copy[1] = 5.0;
        assert(dv_orig[1] == 2.0); // Original unchanged

        // Complex number tests
        {
            alias C = Complex!double;

            // 2D complex vector
            auto cv2 = Vector2D!C(C(1, 1), C(2, 2));
            assert(cv2.magnitudeSquared().isClose(10.0), "Complex vector magnitude squared failed, expected: 10.0, got: " ~ to!string(
                    cv2.magnitudeSquared())); // |1+i|² + |2+2i|² = 2 + 8 = 10

            // Test dot product with complex numbers
            auto cv2b = Vector2D!C(C(1, 0), C(0, 1));
            auto dot_result = cv2b.dot(cv2b);
            assert(dot_result.re.isClose(2.0), "Complex dot product real part failed, expected: 2.0, got: " ~ to!string(
                    dot_result.re)); // (1-0i)(1+0i) + (0-i)(0+i) = 1 + 1 = 2
            assert(dot_result.im.isClose(0.0), "Complex dot product imaginary part failed, expected: 0.0, got: " ~ to!string(
                    dot_result.im));

            // Test unit vector with complex components
            auto cv2_unit = cv2.unit();
            assert((cv2_unit.magnitude() - 1.0).abs < 1e-10, "Complex unit vector magnitude deviation: " ~ to!string(
                    (cv2_unit.magnitude() - 1.0).abs));

            // Dynamic complex vector
            auto dcv = DynamicVector!C([C(1, 1), C(2, 2), C(3, 3)]);
            assert(dcv.length == 3, "Dynamic complex vector dimension failed, expected: 3, got: " ~ to!string(
                    dcv.length));

            // Test Hadamard operations
            auto cv2c = Vector2D!C(C(1, 1), C(2, 2));
            auto cv2d = Vector2D!C(C(2, 0), C(3, 0));
            auto hadamard_prod = cv2c.hadamard(cv2d);
            assert(hadamard_prod[0] == C(2, 2), "Complex Hadamard product[0] failed, expected: 2+2i, got: " ~ to!string(
                    hadamard_prod[0]));
            assert(hadamard_prod[1] == C(6, 6), "Complex Hadamard product[1] failed, expected: 6+6i, got: " ~ to!string(
                    hadamard_prod[1]));

            // Test arithmetic operations
            auto complexSum = cv2c + cv2d;
            assert(complexSum[0] == C(3, 1), "Complex vector addition[0] failed, expected: 3+i, got: " ~ to!string(
                    complexSum[0]));
            assert(complexSum[1] == C(5, 2), "Complex vector addition[1] failed, expected: 5+2i, got: " ~ to!string(
                    complexSum[1]));

            auto complexScaled = cv2c * C(2, 1); // Scale by 2+i
            assert(complexScaled[0] == C(1, 3), "Complex scalar multiplication[0] failed, expected: 1+3i, got: " ~ to!string(
                    complexScaled[0]));
            assert(complexScaled[1] == C(2, 6), "Complex scalar multiplication[1] failed, expected: 2+6i, got: " ~ to!string(
                    complexScaled[1]));
        }

        // Double SIMD tests
        {
            // SIMD-specific tests for double2/double4
            static if (is(double2))
            {
                auto dv2_simd = Vector2D!double(2.0, 3.0);
                auto dv2b_simd = Vector2D!double(4.0, 5.0);

                // Test SIMD operations
                auto simd_sum = dv2_simd + dv2b_simd;
                assert(abs(simd_sum[0] - 6.0) < double.epsilon &&
                        abs(simd_sum[1] - 8.0) < double.epsilon,
                        "SIMD vector addition failed, expected: [6.0, 8.0], got: [" ~ to!string(
                            simd_sum[0]) ~ ", " ~ to!string(simd_sum[1]) ~ "]");

                auto simd_mul = dv2_simd * 2.0;
                assert(abs(simd_mul[0] - 4.0) < double.epsilon &&
                        abs(simd_mul[1] - 6.0) < double.epsilon,
                        "SIMD scalar multiplication failed, expected: [4.0, 6.0], got: [" ~ to!string(
                            simd_mul[0]) ~ ", " ~ to!string(simd_mul[1]) ~ "]");

                auto simd_div = dv2_simd / 2.0;
                assert(abs(simd_div[0] - 1.0) < double.epsilon &&
                        abs(simd_div[1] - 1.5) < double.epsilon,
                        "SIMD scalar division failed, expected: [1.0, 1.5], got: [" ~ to!string(
                            simd_div[0]) ~ ", " ~ to!string(simd_div[1]) ~ "]");

                auto simd_dot = dv2_simd.dot(dv2b_simd);
                assert(abs(simd_dot - 23.0) < double.epsilon,
                    "SIMD dot product failed, expected: 23.0, got: " ~ to!string(simd_dot));

                // Test hadamard operations
                auto hadamard = dv2_simd.hadamard(dv2b_simd);
                assert(abs(hadamard[0] - 8.0) < double.epsilon &&
                        abs(hadamard[1] - 15.0) < double.epsilon,
                        "SIMD Hadamard multiplication failed, expected: [8.0, 15.0], got: [" ~ to!string(
                            hadamard[0]) ~ ", " ~ to!string(hadamard[1]) ~ "]");
            }

            // Test double3 with SIMD (if available)
            static if (is(double4))
            {
                auto dv3_simd = Vector3D(1.0, 2.0, 3.0);
                auto dv3b_simd = Vector3D(4.0, 5.0, 6.0);

                auto simd_sum3 = dv3_simd + dv3b_simd;
                assert(abs(simd_sum3[0] - 5.0) < double.epsilon &&
                        abs(simd_sum3[1] - 7.0) < double.epsilon &&
                        abs(simd_sum3[2] - 9.0) < double.epsilon,
                        "SIMD 3D vector addition failed, expected: [5.0, 7.0, 9.0], got: [" ~
                        to!string(
                            simd_sum3[0]) ~ ", " ~ to!string(
                            simd_sum3[1]) ~ ", " ~ to!string(simd_sum3[2]) ~ "]");

                auto simd_mul3 = dv3_simd * 2.0;
                assert(abs(simd_mul3[0] - 2.0) < double.epsilon &&
                        abs(simd_mul3[1] - 4.0) < double.epsilon &&
                        abs(simd_mul3[2] - 6.0) < double.epsilon,
                        "SIMD 3D scalar multiplication failed, expected: [2.0, 4.0, 6.0], got: [" ~
                        to!string(
                            simd_mul3[0]) ~ ", " ~ to!string(
                            simd_mul3[1]) ~ ", " ~ to!string(simd_mul3[2]) ~ "]");

                auto simd_dot3 = dv3_simd.dot(dv3b_simd);
                assert(abs(simd_dot3 - 32.0) < double.epsilon,
                    "SIMD 3D dot product failed, expected: 32.0, got: " ~ to!string(simd_dot3)); // 1*4 + 2*5 + 3*6 = 32

                // Test SIMD opOpAssign
                auto v3_assign = Vector3D(1.0, 2.0, 3.0);
                v3_assign += Vector3D(2.0, 3.0, 4.0);
                assert(abs(v3_assign[0] - 3.0) < double.epsilon &&
                        abs(v3_assign[1] - 5.0) < double.epsilon &&
                        abs(v3_assign[2] - 7.0) < double.epsilon,
                        "SIMD vector opOpAssign('+') failed");

                v3_assign *= 2.0;
                assert(abs(v3_assign[0] - 6.0) < double.epsilon &&
                        abs(v3_assign[1] - 10.0) < double.epsilon &&
                        abs(v3_assign[2] - 14.0) < double.epsilon,
                        "SIMD vector opOpAssign('*') failed");

                v3_assign /= 2.0;
                assert(abs(v3_assign[0] - 3.0) < double.epsilon &&
                        abs(v3_assign[1] - 5.0) < double.epsilon &&
                        abs(v3_assign[2] - 7.0) < double.epsilon,
                        "SIMD vector opOpAssign('/') failed");
            }

            // Integer vector tests
            {
                auto iv2 = Vector2D!int(3, 4);
                assert(iv2.magnitude() == 5, "Integer vector magnitude failed, expected: 5, got: " ~ to!string(
                        iv2.magnitude()));

                auto iv3 = Vector3D!int(2, 3, 6);
                assert(iv3.dot(iv3) == 49, "Integer vector dot product failed, expected: 49, got: " ~ to!string(
                        iv3.dot(iv3)));

                // Test arithmetic
                auto iv2b = Vector2D!int(1, 2);
                auto intSum = iv2 + iv2b;
                assert(intSum[0] == 4, "Integer vector addition[0] failed, expected: 4, got: " ~ to!string(
                        intSum[0]));
                assert(intSum[1] == 6, "Integer vector addition[1] failed, expected: 6, got: " ~ to!string(
                        intSum[1]));

                auto intScaled = iv2 * 2;
                assert(intScaled[0] == 6, "Integer vector scalar multiplication[0] failed, expected: 6, got: " ~ to!string(
                        intScaled[0]));
                assert(intScaled[1] == 8, "Integer vector scalar multiplication[1] failed, expected: 8, got: " ~ to!string(
                        intScaled[1]));
            }

            // Float vector tests with SIMD
            {
                // Basic float vector tests
                auto fv2 = Vector2D!float(3.0f, 4.0f);
                assert(abs(fv2.magnitude() - 5.0f) < float.epsilon,
                    "Float vector magnitude failed, expected: 5.0, got: " ~ to!string(
                        fv2.magnitude()));

                auto fv2b = Vector2D!float(1.0f, 2.0f);
                auto sum = fv2 + fv2b;
                assert(sum[0] == 4.0f, "Float vector addition[0] failed, expected: 4.0, got: " ~ to!string(
                        sum[0]));
                assert(sum[1] == 6.0f, "Float vector addition[1] failed, expected: 6.0, got: " ~ to!string(
                        sum[1]));

                // SIMD-specific tests for float2/float4
                static if (is(float2))
                {
                    auto fv2_simd = Vector2D!float(2.0f, 3.0f);
                    auto fv2b_simd = Vector2D!float(4.0f, 5.0f);

                    // Test SIMD operations
                    auto simd_sum = fv2_simd + fv2b_simd;
                    assert(simd_sum[0] == 6.0f && simd_sum[1] == 8.0f,
                        "Float SIMD vector addition failed, expected: [6.0, 8.0], got: [" ~ to!string(
                            simd_sum[0]) ~ ", " ~ to!string(simd_sum[1]) ~ "]");

                    auto simd_mul = fv2_simd * 2.0f;
                    assert(simd_mul[0] == 4.0f && simd_mul[1] == 6.0f,
                        "Float SIMD scalar multiplication failed, expected: [4.0, 6.0], got: [" ~ to!string(
                            simd_mul[0]) ~ ", " ~ to!string(simd_mul[1]) ~ "]");

                    auto simd_div = fv2_simd / 2.0f;
                    assert(simd_div[0] == 1.0f && simd_div[1] == 1.5f,
                        "Float SIMD scalar division failed, expected: [1.0, 1.5], got: [" ~ to!string(
                            simd_div[0]) ~ ", " ~ to!string(simd_div[1]) ~ "]");

                    auto simd_dot = fv2_simd.dot(fv2b_simd);
                    assert(simd_dot == 23.0f,
                        "Float SIMD dot product failed, expected: 23.0, got: " ~ to!string(
                            simd_dot)); // 2*4 + 3*5 = 23
                }

                // Test float3 with SIMD (if available)
                static if (is(float4))
                {
                    auto fv3_simd = Vector3D!float(1.0f, 2.0f, 3.0f);
                    auto fv3b_simd = Vector3D!float(4.0f, 5.0f, 6.0f);

                    auto simd_sum3 = fv3_simd + fv3b_simd;
                    assert(simd_sum3[0] == 5.0f && simd_sum3[1] == 7.0f && simd_sum3[2] == 9.0f,
                        "Float SIMD 3D vector addition failed, expected: [5.0, 7.0, 9.0], got: [" ~
                            to!string(
                                simd_sum3[0]) ~ ", " ~ to!string(
                                simd_sum3[1]) ~ ", " ~ to!string(simd_sum3[2]) ~ "]");

                    auto simd_mul3 = fv3_simd * 2.0f;
                    assert(simd_mul3[0] == 2.0f && simd_mul3[1] == 4.0f && simd_mul3[2] == 6.0f,
                        "Float SIMD 3D scalar multiplication failed, expected: [2.0, 4.0, 6.0], got: [" ~
                            to!string(
                                simd_mul3[0]) ~ ", " ~ to!string(
                                simd_mul3[1]) ~ ", " ~ to!string(simd_mul3[2]) ~ "]");

                    auto simd_dot3 = fv3_simd.dot(fv3b_simd);
                    assert(simd_dot3 == 32.0f,
                        "Float SIMD 3D dot product failed, expected: 32.0, got: " ~ to!string(
                            simd_dot3)); // 1*4 + 2*5 + 3*6 = 32
                }

                // Hadamard operations with SIMD
                static if (is(float2))
                {
                    auto hv2a = Vector2D!float(2.0f, 3.0f);
                    auto hv2b = Vector2D!float(4.0f, 5.0f);

                    auto hadamard = hv2a.hadamard(hv2b);
                    assert(hadamard[0] == 8.0f && hadamard[1] == 15.0f,
                        "Float SIMD Hadamard multiplication failed, expected: [8.0, 15.0], got: [" ~ to!string(
                            hadamard[0]) ~ ", " ~ to!string(hadamard[1]) ~ "]");

                    auto hadamard_div = hv2a.hadamardDiv(hv2b);
                    assert(abs(hadamard_div[0] - 0.5f) < float.epsilon,
                        "Float SIMD Hadamard division[0] failed, expected: 0.5, got: " ~ to!string(
                            hadamard_div[0]));
                    assert(abs(hadamard_div[1] - 0.6f) < float.epsilon,
                        "Float SIMD Hadamard division[1] failed, expected: 0.6, got: " ~ to!string(
                            hadamard_div[1]));
                }
            }

            // Test zero methods
            {
                // Test compile-time dimension zero vectors
                {
                    // Vector1D zero
                    auto v1z = Vector1D!double.zero();
                    assert(v1z[0] == 0.0, "Vector1D zero failed");

                    // Vector2D zero
                    auto v2z = Vector2D!double.zero();
                    assert(v2z[0] == 0.0 && v2z[1] == 0.0, "Vector2D zero failed");

                    // Vector3D zero
                    auto v3z = Vector3D!double.zero();
                    assert(v3z[0] == 0.0 && v3z[1] == 0.0 && v3z[2] == 0.0, "Vector3D zero failed");
                }

                // Test runtime dimension zero vectors
                {
                    // Zero-length dynamic vector
                    auto dv0 = DynamicVector!double.zero(0);
                    assert(dv0.length == 0, "Zero-length dynamic vector failed");

                    // Normal dynamic vector
                    auto dv3 = DynamicVector!double.zero(3);
                    assert(dv3.length == 3, "Dynamic vector dimension mismatch");
                    foreach (i; 0 .. 3)
                    {
                        assert(dv3[i] == 0.0, "Dynamic vector component " ~ to!string(
                                i) ~ " not zero");
                    }
                }

                // Test complex number zero vectors
                {
                    alias C = Complex!double;

                    // Complex Vector2D zero
                    auto cv2 = Vector2D!C.zero();
                    assert(cv2[0] == C(0, 0), "Complex Vector2D zero[0] failed");
                    assert(cv2[1] == C(0, 0), "Complex Vector2D zero[1] failed");

                    // Complex dynamic vector zero
                    auto cdv = DynamicVector!C.zero(2);
                    assert(cdv.length == 2, "Complex dynamic vector dimension mismatch");
                    assert(cdv[0] == C(0, 0), "Complex dynamic vector zero[0] failed");
                    assert(cdv[1] == C(0, 0), "Complex dynamic vector zero[1] failed");
                }

                // Test integer zero vectors
                {
                    // Integer Vector3D zero
                    auto iv3 = Vector3D!int.zero();
                    assert(iv3[0] == 0 && iv3[1] == 0 && iv3[2] == 0, "Integer Vector3D zero failed");

                    // Integer dynamic vector zero
                    auto idv = DynamicVector!int.zero(2);
                    assert(idv.length == 2, "Integer dynamic vector dimension mismatch");
                    assert(idv[0] == 0 && idv[1] == 0, "Integer dynamic vector zero components failed");
                }

                // Test float zero vectors
                {
                    // Float Vector2D zero
                    auto fv2 = Vector2D!float.zero();
                    assert(fv2[0] == 0.0f && fv2[1] == 0.0f, "Float Vector2D zero failed");

                    // Float dynamic vector zero
                    auto fdv = DynamicVector!float.zero(3);
                    assert(fdv.length == 3, "Float dynamic vector dimension mismatch");
                    foreach (i; 0 .. 3)
                    {
                        assert(fdv[i] == 0.0f, "Float dynamic vector component " ~ to!string(
                                i) ~ " not zero");
                    }
                }

                // Test dimension matching for compile-time vectors
                {
                    bool threw = false;
                    try
                    {
                        auto v2 = Vector2D!double.zero(3); // Should throw - dimension mismatch
                    }
                    catch (Error e)
                    {
                        threw = true;
                    }
                    assert(threw, "Should throw on dimension mismatch");
                }
            }

            // Test compile-time instantiation
            {
                // Test struct with compile-time Vector field initialization
                struct TestConfig(V)
                {
                    V field = V.zero();
                }

                // Also verify the values are correct
                TestConfig!(Vector1D!double) test1;
                assert(test1.field[0] == 0.0);

                TestConfig!(Vector2D!double) test2;
                assert(test2.field[0] == 0.0 && test2.field[1] == 0.0);

                TestConfig!(Vector3D!double) test3;
                assert(test3.field[0] == 0.0 && test3.field[1] == 0.0 && test3.field[2] == 0.0);

                TestConfig!(DynamicVector!double) test4;
                assert(test4.field.length == 0);
            }

            // Test serialization/deserialization

            // Test canary value validation
            {
                // Test invalid canary value
                auto v1_bad = Vector1D!double(3.14);
                auto v1_bytes_bad = v1_bad.toBytes();
                v1_bytes_bad[0] = 42; // Corrupt canary value
                bool threw = false;
                try
                {
                    auto v1_restored = Vector1D!double.fromBytes(v1_bytes_bad);
                }
                catch (Error e)
                {
                    threw = true;
                    assert(e.msg == "Invalid vector data - incorrect canary value");
                }
                assert(threw, "Should throw on invalid canary");
            }

            // Test serialization of vectors
            {
                // Test empty data
                {
                    bool threw = false;
                    try
                    {
                        auto v_empty = Vector1D!double.fromBytes([]);
                    }
                    catch (Error e)
                    {
                        threw = true;
                        assert(e.msg == "Data too short for canary value");
                    }
                    assert(threw, "Should throw on empty data");
                }

                // Fixed dimension vectors
                {
                    auto v1_ser = Vector1D!double(3.14);
                    auto v1_bytes = v1_ser.toBytes();
                    auto v1_restored = Vector1D!double.fromBytes(v1_bytes);
                    assert(v1_ser == v1_restored, format("1D vector serialization/deserialization failed\nExpected: %s\nGot: %s",
                            v1_ser, v1_restored));

                    auto v2_ser = Vector2D!double(1.23, 4.56);
                    auto v2_bytes = v2_ser.toBytes();
                    auto v2_restored = Vector2D!double.fromBytes(v2_bytes);
                    assert(v2_ser == v2_restored, format("2D vector serialization/deserialization failed\nExpected: %s\nGot: %s",
                            v2_ser, v2_restored));

                    auto v3_ser = Vector3D!double(1.1, 2.2, 3.3);
                    auto v3_bytes = v3_ser.toBytes();
                    auto v3_restored = Vector3D!double.fromBytes(v3_bytes);
                    assert(v3_ser == v3_restored, format("3D vector serialization/deserialization failed\nExpected: %s\nGot: %s",
                            v3_ser, v3_restored));
                }

                // Dynamic vector
                {
                    auto dv_ser = DynamicVector!double([1.1, 2.2, 3.3, 4.4]);
                    auto dv_bytes = dv_ser.toBytes();
                    auto dv_restored = DynamicVector!double.fromBytes(dv_bytes);
                    assert(dv_ser == dv_restored, format("Dynamic vector serialization/deserialization failed\nExpected: %s\nGot: %s",
                            dv_ser.toString(), dv_restored.toString()));
                    assert(dv_ser.length == dv_restored.length,
                        "Dynamic vector dimension mismatch after serialization, expected: " ~ to!string(
                            dv_ser.length) ~
                            ", got: " ~ to!string(dv_restored.length));
                }

                // Test reference fromBytes
                {
                    auto v1_ser = Vector1D!double(3.14);
                    auto v1_bytes = v1_ser.toBytes();
                    Vector1D!double v1_ref;
                    size_t v1_bytes_read = Vector1D!double.fromBytes(v1_bytes, v1_ref);
                    assert(v1_bytes_read == 1 + long.sizeof + double.sizeof,
                        "1D vector bytes read mismatch, expected: " ~ to!string(
                            1 + long.sizeof + double.sizeof) ~ ", got: " ~ to!string(v1_bytes_read));
                    assert(v1_ref == v1_ser, format(
                            "1D vector reference deserialization failed\nExpected: %s\nGot: %s",
                            v1_ser, v1_ref));

                    auto v2_ser = Vector2D!double(1.23, 4.56);
                    auto v2_bytes = v2_ser.toBytes();
                    Vector2D!double v2_ref;
                    size_t v2_bytes_read = Vector2D!double.fromBytes(v2_bytes, v2_ref);
                    assert(v2_bytes_read == 1 + long.sizeof + double.sizeof * 2,
                        "2D vector bytes read mismatch, expected: " ~ to!string(
                            1 + long.sizeof + double.sizeof * 2) ~ ", got: " ~ to!string(
                            v2_bytes_read));
                    assert(v2_ref == v2_ser, format(
                            "2D vector reference deserialization failed\nExpected: %s\nGot: %s",
                            v2_ser, v2_ref));

                    auto v3_ser = Vector3D!double(1.1, 2.2, 3.3);
                    auto v3_bytes = v3_ser.toBytes();
                    Vector3D!double v3_ref;
                    size_t v3_bytes_read = Vector3D!double.fromBytes(v3_bytes, v3_ref);
                    assert(v3_bytes_read == 1 + long.sizeof + double.sizeof * 3,
                        "3D vector bytes read mismatch, expected: " ~ to!string(
                            1 + long.sizeof + double.sizeof * 3) ~ ", got: " ~ to!string(
                            v3_bytes_read));
                    assert(v3_ref == v3_ser, format(
                            "3D vector reference deserialization failed\nExpected: %s\nGot: %s",
                            v3_ser, v3_ref));

                    auto dv_ser = DynamicVector!double([1.1, 2.2, 3.3, 4.4]);
                    auto dv_bytes = dv_ser.toBytes();
                    DynamicVector!double dv_ref;
                    size_t dv_bytes_read = DynamicVector!double.fromBytes(dv_bytes, dv_ref);
                    assert(dv_bytes_read == 1 + long.sizeof + double.sizeof * dv_ser.length,
                        "Dynamic vector bytes read mismatch, expected: " ~ to!string(
                            1 + size_t.sizeof + double.sizeof * dv_ser.length) ~
                            ", got: " ~ to!string(
                                dv_bytes_read));
                    assert(dv_ref == dv_ser, format(
                            "Dynamic vector reference deserialization failed\nExpected: %s\nGot: %s",
                            dv_ser, dv_ref));
                    assert(dv_ref.length == dv_ser.length,
                        "Dynamic vector reference dimension mismatch, expected: " ~ to!string(
                            dv_ser.length) ~
                            ", got: " ~ to!string(dv_ref.length));
                }
            }
        }
    }
}

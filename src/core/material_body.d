module core.material_body;

import core.material_point;
import math.vector;

// Generic MaterialBody class
class MaterialBody(T, V) if (isMaterialPoint!(T, V)) {
    private T[] points;
    private double horizon;
    
    // Constructor
    this(T[] initialPoints, double horizonRadius) {
        points = initialPoints;
        horizon = horizonRadius;
    }
    
    // Get neighbors within horizon based on current positions
    const(T)[] neighbors(size_t index) const {
        const(T)[] result;
        V pos = points[index].position;
        
        for (size_t i = 0; i < points.length; i++) {
            if (i == index) continue;
            
            V neighborPos = points[i].position;
            double distance = (neighborPos - pos).magnitude();
            
            if (distance <= horizon) {
                result ~= points[i];
            }
        }
        
        return result;
    }
    
    // Number of points
    @property size_t numPoints() const {
        return points.length;
    }
    
    // Indexing operators
    const(T) opIndex(size_t index) const {
        return points[index];
    }
    
    T opIndex(size_t index) {
        return points[index];
    }
    
    void opIndexAssign(T point, size_t index) {
        points[index] = point;
    }
}

// Unit tests
unittest {
    import std.math : abs;
    
    // Create a simple test material point implementation
    class TestPoint(V) : MaterialPoint!(TestPoint!V, V) {
        private V _pos;
        private V _refPos;
        
        this(V pos) {
            _pos = pos;
            _refPos = pos;
        }
        
        @property V position() const { return _pos; }
        @property V referencePosition() const { return _refPos; }
        
        TestPoint!V opBinary(string op)(TestPoint!V other) const
            if (op == "+") {
            return new TestPoint!V(_pos + other._pos);
        }
        
        TestPoint!V opBinary(string op)(double scalar) const
            if (op == "*") {
            return new TestPoint!V(_pos * scalar);
        }
        
        TestPoint!V stateRate(TestPoint!V[] neighbors) const {
            return new TestPoint!V(V.zero());
        }
    }
    
    // Test with 2D points
    auto p1 = new TestPoint!Vector2D(Vector2D(0.0, 0.0));
    auto p2 = new TestPoint!Vector2D(Vector2D(1.0, 0.0));
    auto p3 = new TestPoint!Vector2D(Vector2D(2.0, 0.0));
    auto p4 = new TestPoint!Vector2D(Vector2D(0.0, 2.0));
    
    auto points = [p1, p2, p3, p4];
    auto body = new MaterialBody!(TestPoint!Vector2D, Vector2D)(points, 1.5);
    
    // Test neighbor search
    auto neighbors = body.neighbors(0);
    assert(neighbors.length == 1);  // Only p2 should be within horizon of p1
    
    // Test indexing
    assert(body[0] == p1);
    assert(body.numPoints == 4);
}

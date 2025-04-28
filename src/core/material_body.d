module core.material_body;

import core.material_point;
import math.vector;
import std.stdio;
import std.string : strip;
import std.array : split;

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

    // Export points to CSV file with positions
    void exportToCSV(string filename) const {
        auto f = File(filename, "w");
        
        // Write header
        static if (V.components.length == 1) {
            f.writeln("x,x_ref,vx");
        } else static if (V.components.length == 2) {
            f.writeln("x,y,x_ref,y_ref,vx,vy");
        } else static if (V.components.length == 3) {
            f.writeln("x,y,z,x_ref,y_ref,z_ref,vx,vy,vz");
        }
        
        // Write data rows
        foreach (point; points) {
            auto pos = point.position;
            auto ref_pos = point.referencePosition;
            
            static if (V.components.length == 1) {
                auto vel = point.velocity;
                f.writefln("%g,%g,%g", 
                    pos[0], ref_pos[0], vel[0]);
            } else static if (V.components.length == 2) {
                auto vel = point.velocity;
                f.writefln("%g,%g,%g,%g,%g,%g",
                    pos[0], pos[1], 
                    ref_pos[0], ref_pos[1],
                    vel[0], vel[1]);
            } else static if (V.components.length == 3) {
                auto vel = point.velocity;
                f.writefln("%g,%g,%g,%g,%g,%g,%g,%g,%g",
                    pos[0], pos[1], pos[2],
                    ref_pos[0], ref_pos[1], ref_pos[2],
                    vel[0], vel[1], vel[2]);
            }
        }
    }
}

// Unit tests
unittest {
    import std.math : abs;
    
    // Create a simple test material point implementation
    class TestPoint(V) : MaterialPoint!(TestPoint!V, V) {
        private V _pos;
        private V _refPos;
        private V _vel;
        
        this(V pos, V vel = V.zero()) {
            _pos = pos;
            _refPos = pos;
            _vel = vel;
        }
        
        @property V position() const { return _pos; }
        @property V referencePosition() const { return _refPos; }
        @property V velocity() const { return _vel; }
        
        TestPoint!V opBinary(string op)(TestPoint!V other) const
            if (op == "+") {
            return new TestPoint!V(_pos + other._pos);
        }
        
        TestPoint!V opBinary(string op)(double scalar) const
            if (op == "*") {
            return new TestPoint!V(_pos * scalar);
        }
        
        void updateState(const(TestPoint!V)[] neighbors, double timeStep) {
            // Simple test implementation - no state changes
        }
    }
    
    // Test with 2D points
    auto p1 = new TestPoint!Vector2D(Vector2D(0.0, 0.0), Vector2D(1.0, 0.0));
    auto p2 = new TestPoint!Vector2D(Vector2D(1.0, 0.0), Vector2D(0.0, 1.0));
    auto p3 = new TestPoint!Vector2D(Vector2D(2.0, 0.0), Vector2D(-1.0, 0.0));
    auto p4 = new TestPoint!Vector2D(Vector2D(0.0, 2.0), Vector2D(0.0, -1.0));
    
    auto points = [p1, p2, p3, p4];
    auto body = new MaterialBody!(TestPoint!Vector2D, Vector2D)(points, 1.5);
    
    // Test CSV export
    import std.file : exists, remove;
    body.exportToCSV("test_points.csv");
    
    // Verify file was created and has correct format
    assert(exists("test_points.csv"));
    auto f = File("test_points.csv");
    
    // Check header
    string header = f.readln().strip();
    assert(header == "x,y,x_ref,y_ref,vx,vy", "Incorrect CSV header");
    
    // Check first data row (point at 0,0 with velocity 1,0)
    auto firstRow = f.readln().strip().split(",");
    assert(firstRow.length == 6, "Incorrect number of columns");
    assert(firstRow[0] == "0", "Incorrect x position");
    assert(firstRow[1] == "0", "Incorrect y position");
    assert(firstRow[2] == "0", "Incorrect x reference");
    assert(firstRow[3] == "0", "Incorrect y reference");
    assert(firstRow[4] == "1", "Incorrect x velocity");
    assert(firstRow[5] == "0", "Incorrect y velocity");
    
    // Cleanup
    f.close();
    remove("test_points.csv");
    
    // Test neighbor search
    auto neighbors = body.neighbors(0);
    assert(neighbors.length == 1);  // Only p2 should be within horizon of p1
    
    // Test indexing
    assert(body[0] == p1);
    assert(body.numPoints == 4);
}

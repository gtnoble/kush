module core.material_body;

import core.material_point;
import math.vector;
import std.stdio;
import std.string : strip;
import std.array : split, appender;
import std.math : sqrt;

// Generic MaterialBody class
class MaterialBody(T, V) if (isMaterialPoint!(T, V)) {
    private struct NeighborCache {
        T[] neighbors;     // List of neighboring points
        bool isValid;      // Flag indicating if cache is valid
    }

    private T[] points;
    private double horizon;
    private NeighborCache[] neighborCache;  // Cache for each point's neighbors
    
    // Spatial grid based on dimension
    static if (V.components.length == 1) {
        private size_t[][long] grid;      // 1D: x -> [point indices]
    }
    else static if (V.components.length == 2) {
        private size_t[][long][long] grid; // 2D: x -> y -> [point indices]
    }
    else static if (V.components.length == 3) {
        private size_t[][long][long][long] grid; // 3D: x -> y -> z -> [point indices]
    }
    
    private double cellSize;
    private bool gridNeedsUpdate = true;
    
    // Constructor
    this(T[] initialPoints, double horizonRadius) {
        points = initialPoints;
        horizon = horizonRadius;
        neighborCache = new NeighborCache[points.length];
        foreach (ref cache; neighborCache) {
            cache.isValid = false;
        }
    }
    
    // Calculate optimal cell size based on dimension
    private double getOptimalCellSize() const {
        static if (V.components.length == 1)
            return horizon;  // 1D case
        else static if (V.components.length == 2)
            return horizon / sqrt(2.0);  // 2D optimal
        else static if (V.components.length == 3)
            return horizon / sqrt(3.0);  // 3D optimal
    }
    
    // Update spatial grid
    private void updateGrid() {
        // Invalidate all caches as spatial relationships will change
        foreach (ref cache; neighborCache) {
            cache.isValid = false;
        }

        // Clear existing grid
        static if (V.components.length == 1) {
            grid = null;
        }
        else static if (V.components.length == 2) {
            grid = null;
        }
        else static if (V.components.length == 3) {
            grid = null;
        }
        
        cellSize = getOptimalCellSize();
        
        // Add points to grid
        foreach (size_t i, point; points) {
            auto pos = point.position;
            long x = cast(long)(pos[0] / cellSize);
            
            static if (V.components.length == 1) {
                if (x !in grid) grid[x] = [];
                grid[x] ~= i;
            }
            else static if (V.components.length == 2) {
                long y = cast(long)(pos[1] / cellSize);
                if (x !in grid) grid[x] = typeof(grid[x]).init;
                if (y !in grid[x]) grid[x][y] = [];
                grid[x][y] ~= i;
            }
            else static if (V.components.length == 3) {
                long y = cast(long)(pos[1] / cellSize);
                long z = cast(long)(pos[2] / cellSize);
                if (x !in grid) grid[x] = typeof(grid[x]).init;
                if (y !in grid[x]) grid[x][y] = typeof(grid[x][y]).init;
                if (z !in grid[x][y]) grid[x][y][z] = [];
                grid[x][y][z] ~= i;
            }
        }
        
        gridNeedsUpdate = false;
    }
    
    // Helper to get points in a cell
    private const(size_t)[] getPointsInCell(long x, long y = 0, long z = 0) const {
        static if (V.components.length == 1) {
            return (x in grid) ? grid[x] : [];
        }
        else static if (V.components.length == 2) {
            return (x in grid && y in grid[x]) ? grid[x][y] : [];
        }
        else static if (V.components.length == 3) {
            return (x in grid && y in grid[x] && z in grid[x][y]) ? 
                grid[x][y][z] : [];
        }
    }
    
    // Get neighbors within horizon based on current positions
    void neighbors(size_t index, ref T[] result) {
        if (gridNeedsUpdate) {
            updateGrid();
        }

        // Return cached neighbors if valid
        if (neighborCache[index].isValid) {
            result = neighborCache[index].neighbors;
            return;
        }

        // Calculate neighbors if cache is invalid
        result.length = 0;
        auto resultAppender = appender(&result);

        V pos = points[index].position;
        
        // Get cell coordinates for current point
        long x = cast(long)(pos[0] / cellSize);
        
        static if (V.components.length == 1) {
            // Check neighboring cells in 1D
            foreach (dx; -1..2) {
                foreach (neighborIndex; getPointsInCell(x + dx)) {
                    if (neighborIndex == index) continue;
                    
                    if ((points[neighborIndex].position - pos).magnitudeSquared() <= horizon ^^ 2) {
                        resultAppender.put(points[neighborIndex]);
                    }
                }
            }

            // Store in cache
            neighborCache[index].neighbors = result.dup;
            neighborCache[index].isValid = true;
        }
        else static if (V.components.length == 2) {
            long y = cast(long)(pos[1] / cellSize);
            
            // Check neighboring cells in 2D
            foreach (dx; -1..2)
            foreach (dy; -1..2) {
                auto neighborIndices = getPointsInCell(x + dx, y + dy);
                foreach (neighborIndex; neighborIndices) {
                    if (neighborIndex == index) continue;
                    
                    if ((points[neighborIndex].position - pos).magnitudeSquared() <= horizon ^^ 2) {
                        resultAppender.put(points[neighborIndex]);
                    }
                }
            }

            // Store in cache
            neighborCache[index].neighbors = result.dup;
            neighborCache[index].isValid = true;
        }
        else static if (V.components.length == 3) {
            long y = cast(long)(pos[1] / cellSize);
            long z = cast(long)(pos[2] / cellSize);
            
            // Check neighboring cells in 3D
            foreach (dx; -1..2)
            foreach (dy; -1..2)
            foreach (dz; -1..2) {
                foreach (neighborIndex; getPointsInCell(x + dx, y + dy, z + dz)) {
                    if (neighborIndex == index) continue;
                    
                    if ((points[neighborIndex].position - pos).magnitudeSquared() <= horizon ^^ 2) {
                        resultAppender.put(points[neighborIndex]);
                    }
                }
            }

            // Store in cache
            neighborCache[index].neighbors = result.dup;
            neighborCache[index].isValid = true;
        }
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
        // When a point moves, invalidate its cache and the caches of potential neighbors
        points[index] = point;
        gridNeedsUpdate = true;

        // Invalidate cache for the moved point
        neighborCache[index].isValid = false;

        V pos = point.position;
        long x = cast(long)(pos[0] / cellSize);
        
        static if (V.components.length == 1) {
            foreach (dx; -1..2) {
                foreach (neighborIndex; getPointsInCell(x + dx)) {
                    if (neighborIndex != index) {
                        neighborCache[neighborIndex].isValid = false;
                    }
                }
            }
        }
        else static if (V.components.length == 2) {
            long y = cast(long)(pos[1] / cellSize);
            foreach (dx; -1..2)
            foreach (dy; -1..2) {
                foreach (neighborIndex; getPointsInCell(x + dx, y + dy)) {
                    if (neighborIndex != index) {
                        neighborCache[neighborIndex].isValid = false;
                    }
                }
            }
        }
        else static if (V.components.length == 3) {
            long y = cast(long)(pos[1] / cellSize);
            long z = cast(long)(pos[2] / cellSize);
            foreach (dx; -1..2)
            foreach (dy; -1..2)
            foreach (dz; -1..2) {
                foreach (neighborIndex; getPointsInCell(x + dx, y + dy, z + dz)) {
                    if (neighborIndex != index) {
                        neighborCache[neighborIndex].isValid = false;
                    }
                }
            }
        }
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

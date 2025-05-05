module io.point_loader;

import std.json;
import std.stdio;
import std.conv;
import std.algorithm : map;
import std.array : array;
import std.exception : enforce;
import std.format : format;
import std.sumtype;
import core.material_point;
import math.vector;
import io.material_loader : MaterialsConfig, loadMaterialConfigs;

/// Get dimension from position vector JSON
private int getDimension(JSONValue positionJson) {
    enforce(positionJson.type == JSONType.array,
        "Position must be a JSON array");
        
    auto arr = positionJson.array;
    enforce(arr.length > 0 && arr.length <= 3,
        format("Invalid vector dimension: %d (must be 1, 2, or 3)", arr.length));
        
    return cast(int)arr.length;
}

/// Detect and validate dimension from points file
int detectDimensionFromFile(string filepath) {
    import std.file : exists;
    
    enforce(exists(filepath), "Points file does not exist: " ~ filepath);
    
    auto file = File(filepath, "r");
    int dimension = -1;
    int lineNum = 0;
    
    foreach (line; file.byLine) {
        lineNum++;
        // Skip empty lines
        if (line.length == 0) continue;
        
        // Parse JSON and get position
        JSONValue point;
        try {
            point = parseJSON(line);
        } catch (JSONException e) {
            throw new Exception(
                format("Invalid JSON on line %d: %s", lineNum, e.msg));
        }
        
        enforce("position" in point,
            format("Missing 'position' field on line %d", lineNum));
            
        // Get dimension of current point
        int currentDim = getDimension(point["position"]);
        
        // If this is the first point, set dimension
        if (dimension == -1) {
            dimension = currentDim;
        }
        // Otherwise validate it matches
        else if (currentDim != dimension) {
            throw new Exception(format(
                "Inconsistent dimensions in points file:\n" ~
                "- Line %d: %dD point\n" ~
                "- Line %d: %dD point\n" ~
                "All points must have the same dimension.",
                1, dimension, lineNum, currentDim));
        }
    }
    
    enforce(dimension != -1, "No points found in file: " ~ filepath);
    return dimension;
}

/// Configuration for a point loaded from JSON
struct PointConfig(V) {
    V position;                    // Required: Position in space
    string[] material_groups;      // Required: Array of material group names
    double volume;                 // Required: Volume of the point
}

/// Parse a vector from a JSON array
V parseVector(V)(JSONValue json) {
    import std.exception : enforce;
    import std.format : format;
    
    enforce(json.type == JSONType.array, 
        "Vector must be a JSON array, got %s".format(json.type));
    
    auto arr = json.array;
    enforce(arr.length == V.dimension, 
        "Vector dimension mismatch: expected %d components, got %d"
        .format(V.dimension, arr.length));
    
    static if (V.dimension == 1) {
        return V(arr[0].get!double);
    } else static if (V.dimension == 2) {
        return V(arr[0].get!double, arr[1].get!double);
    } else static if (V.dimension == 3) {
        return V(arr[0].get!double, arr[1].get!double, arr[2].get!double);
    }
}

/// Parse a point configuration from a line of JSON
PointConfig!V parsePointConfig(V)(string jsonLine) {
    import std.exception : enforce;
    
    // Parse JSON line
    JSONValue json = parseJSON(jsonLine);
    
    // Validate required fields exist
    enforce("position" in json, "Missing required field: position");
    enforce("material_groups" in json, "Missing required field: material_groups");
    enforce("volume" in json, "Missing required field: volume");

    PointConfig!V config;
    
    // Parse position vector
    config.position = parseVector!V(json["position"]);
    
    // Parse material groups array
    enforce(json["material_groups"].type == JSONType.array,
        "material_groups must be an array of strings");
    config.material_groups = json["material_groups"]
        .array
        .map!(v => v.get!string)
        .array;
    enforce(config.material_groups.length > 0,
        "At least one material group must be specified");

    // Parse volume
    config.volume = json["volume"].get!double;
    enforce(config.volume > 0.0, "Volume must be positive");
    
    return config;
}

/// Container for loaded points and their material configuration
struct LoadedPoints(V) {
    PointConfig!V[] points;       // The loaded points
    MaterialsConfig!V materials;  // The loaded materials configuration
}

/// Type that can hold points of any dimension
alias DimensionalPoints = SumType!(
    LoadedPoints!Vector1D,
    LoadedPoints!Vector2D,
    LoadedPoints!Vector3D
);

/// Load point configurations with auto-detected dimensionality
DimensionalPoints loadPointConfigsWithDimension(string filepath, string materials_file) {
    // First detect and validate dimension from points
    int dimension = detectDimensionFromFile(filepath);
    
    // Load with appropriate vector type
    final switch (dimension) {
        case 1:
            auto materials = loadMaterialConfigs!Vector1D(materials_file);
            auto points = loadPointConfigs!Vector1D(filepath, materials);
            return DimensionalPoints(LoadedPoints!Vector1D(points, materials));
        case 2:
            auto materials = loadMaterialConfigs!Vector2D(materials_file);
            auto points = loadPointConfigs!Vector2D(filepath, materials);
            return DimensionalPoints(LoadedPoints!Vector2D(points, materials));
        case 3:
            auto materials = loadMaterialConfigs!Vector3D(materials_file);
            auto points = loadPointConfigs!Vector3D(filepath, materials);
            return DimensionalPoints(LoadedPoints!Vector3D(points, materials));
    }
}

/// Load point configurations from a JSONL file and update material group counts
PointConfig!V[] loadPointConfigs(V)(string filepath, ref MaterialsConfig!V materials) {
    import std.exception : enforce;
    import std.file : exists;
    
    enforce(exists(filepath), "Points file does not exist: " ~ filepath);
    
    PointConfig!V[] configs;
    auto file = File(filepath, "r");
    
    // Process file line by line
    int lineNum;
    foreach (line; file.byLine) {
        lineNum++;
        auto config = parsePointConfig!V(line.idup);
        
        // Update point counts for each material group
        foreach (group; config.material_groups) {
            materials.incrementPointCount(group);
        }
        
        configs ~= config;
    }
    
    enforce(configs.length > 0, "No points found in file: " ~ filepath);
    return configs;
}

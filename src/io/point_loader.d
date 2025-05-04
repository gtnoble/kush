module io.point_loader;

import std.json;
import std.stdio;
import std.conv;
import std.algorithm : map;
import std.array : array;
import core.material_point;
import math.vector;
import io.material_loader : MaterialsConfig;

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

/// Load point configurations from a JSONL file and update material group counts
PointConfig!V[] loadPointConfigs(V)(string filepath, ref MaterialsConfig!V materials) {
    import std.exception : enforce;
    import std.file : exists;
    
    enforce(exists(filepath), "Points file does not exist: " ~ filepath);
    
    PointConfig!V[] configs;
    auto file = File(filepath, "r");
    
    // Process file line by line
    foreach (line; file.byLine) {
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

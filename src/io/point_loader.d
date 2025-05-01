module io.point_loader;

import std.json;
import std.stdio;
import std.conv;
import core.material_point;
import math.vector;

/// Configuration for a point loaded from JSON
struct PointConfig(V) {
    V position;
    string material;
    double volume;          // Required: Volume of the point
    V velocity = V.zero();     // Optional: Initial velocity
    bool fixed_velocity;       // Optional: If true, velocity remains constant
    V force = V.zero();       // Optional: Constant force applied to point
    double ramp_duration = 1e-6;  // Optional: Force ramp duration
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
    
    // Required fields
    enforce("position" in json, "Missing required field: position");
    enforce("material" in json, "Missing required field: material");
    
    PointConfig!V config;
    
    // Parse position (required)
    config.position = parseVector!V(json["position"]);
    
    // Parse material reference (required)
    config.material = json["material"].get!string;
    
    // Parse volume (required)
    enforce("volume" in json, "Missing required field: volume");
    config.volume = json["volume"].get!double;
    enforce(config.volume > 0.0, "Volume must be positive");
    
    // Parse optional fields
    if ("velocity" in json) {
        config.velocity = parseVector!V(json["velocity"]);
    }
    
    if ("fixed_velocity" in json) {
        config.fixed_velocity = json["fixed_velocity"].get!bool;
    }
    
    if ("force" in json) {
        config.force = parseVector!V(json["force"]);
    }
    
    if ("ramp_duration" in json) {
        config.ramp_duration = json["ramp_duration"].get!double;
        enforce(config.ramp_duration > 0.0, "Ramp duration must be positive");
    }
    
    return config;
}

/// Load point configurations from a JSONL file
PointConfig!V[] loadPointConfigs(V)(string filepath) {
    import std.exception : enforce;
    import std.file : exists;
    
    enforce(exists(filepath), "Points file does not exist: " ~ filepath);
    
    PointConfig!V[] configs;
    auto file = File(filepath, "r");
    
    // Process file line by line
    foreach (line; file.byLine) {
        configs ~= parsePointConfig!V(line.idup);
    }
    
    enforce(configs.length > 0, "No points found in file: " ~ filepath);
    return configs;
}

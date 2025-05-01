module io.material_loader;

import std.json;
import std.stdio;
import std.math : sqrt;
import std.file : readText;
import std.exception : enforce;
import std.array;
import std.format : format;
import math.vector;

/// Material properties loaded from JSON
struct MaterialConfig(V) {
    // Physical properties (required)
    double density;
    double youngsModulus;
    double criticalStretch;
    
    // Optional properties with defaults
    V velocity = V.zero();
    bool fixed_velocity = false;
    V force = V.zero();
    double ramp_duration = 1e-6;
}

/// Contains all material configurations indexed by name
struct MaterialsConfig(V) {
    MaterialConfig!V[string] materials;
    
    /// Calculate the minimum wave speed across all materials
    double calculateMinWaveSpeed() const {
        import std.algorithm : map, minElement;
        
        // Calculate wave speed for each material: sqrt(E/ρ)
        double[] waveSpeeds = materials.values
            .map!(m => sqrt(m.youngsModulus / m.density))
            .array;
            
        enforce(waveSpeeds.length > 0, "No materials defined");
        return waveSpeeds.minElement;
    }
    
    /// Get a material by name, throw if not found
    ref const(MaterialConfig!V) getMaterial(string name) const {
        enforce(name in materials, "Material not found: " ~ name);
        return materials[name];
    }
}

/// Parse a vector from a JSON array
private V parseVector(V)(JSONValue json) {
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

/// Parse a material configuration from JSON
private MaterialConfig!V parseMaterialConfig(V)(JSONValue json) {
    // Required fields
    enforce("density" in json, "Missing required field: density");
    enforce("youngsModulus" in json, "Missing required field: youngsModulus");
    enforce("criticalStretch" in json, "Missing required field: criticalStretch");
    
    MaterialConfig!V config;
    
    // Parse required fields
    config.density = json["density"].get!double;
    config.youngsModulus = json["youngsModulus"].get!double;
    config.criticalStretch = json["criticalStretch"].get!double;
    
    // Validate required fields
    enforce(config.density > 0.0, "Density must be positive");
    enforce(config.youngsModulus > 0.0, "Young's modulus must be positive");
    enforce(config.criticalStretch > 0.0, "Critical stretch must be positive");
    
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

/// Load material configurations from a JSON file
MaterialsConfig!V loadMaterialConfigs(V)(string filepath) {
    import std.file : exists;
    
    enforce(exists(filepath), "Materials file does not exist: " ~ filepath);
    
    // Read and parse JSON file
    string jsonText = readText(filepath);
    JSONValue json = parseJSON(jsonText);
    
    // Validate top-level structure
    enforce("materials" in json, "Missing required field: materials");
    enforce(json["materials"].type == JSONType.object, 
        "Materials must be an object mapping names to configurations");
    
    MaterialsConfig!V configs;
    
    // Parse each material configuration
    foreach (name, materialJson; json["materials"].object) {
        configs.materials[name] = parseMaterialConfig!V(materialJson);
    }
    
    enforce(configs.materials.length > 0, "No materials found in file: " ~ filepath);
    return configs;
}

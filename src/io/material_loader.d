module io.material_loader;

import std.json;
import std.stdio;
import std.math : sqrt;
import std.file : readText;
import std.exception : enforce;
import std.array;
import std.format : format;
import std.algorithm : map, filter, minElement;
import std.conv : to;
import math.vector;

/// Material properties loaded from JSON
struct MaterialConfig(V) {
    // Physical properties
    double density = 0.0;
    double youngsModulus = 0.0;
    double criticalStretch = 0.0;
    
    // Optional properties with defaults
    V velocity = V.zero();
    bool fixed_velocity = false;
    V force = V.zero();
    double ramp_duration = 1e-6;

    /// Merge this config with another, with the other's properties taking precedence
    MaterialConfig!V merge(const ref MaterialConfig!V other) const {
        MaterialConfig!V result = this;
        
        // Only override non-zero values from other config
        if (other.density > 0.0) result.density = other.density;
        if (other.youngsModulus > 0.0) result.youngsModulus = other.youngsModulus;
        if (other.criticalStretch > 0.0) result.criticalStretch = other.criticalStretch;
        
        // Override optional properties if they differ from defaults
        if (other.velocity != V.zero()) result.velocity = other.velocity;
        if (other.fixed_velocity != false) result.fixed_velocity = other.fixed_velocity;
        if (other.force != V.zero()) result.force = other.force;
        if (other.ramp_duration != 1e-6) result.ramp_duration = other.ramp_duration;
        
        return result;
    }
}

/// Contains all material configurations indexed by name
struct MaterialsConfig(V) {
    MaterialConfig!V[string] materials;
    size_t[string] pointCounts;  // Track number of points per material group
    
    /// Calculate the minimum wave speed across all materials
    double calculateMinWaveSpeed() const {
        import std.algorithm : map, minElement;
        
        // Calculate wave speed for each material: sqrt(E/ρ)
        double[] waveSpeeds = materials.values
            .filter!(m => m.density > 0.0 && m.youngsModulus > 0.0)  // Only consider complete materials
            .map!(m => sqrt(m.youngsModulus / m.density))
            .array;
            
        enforce(waveSpeeds.length > 0, "No complete materials defined with density and Young's modulus");
        return waveSpeeds.minElement;
    }
    
    /// Get a material by name, throw if not found
    ref const(MaterialConfig!V) getMaterial(string name) const {
        enforce(name in materials, "Material group not found: " ~ name);
        return materials[name];
    }

    /// Get scaled force for a material group
    V getScaledForce(string name) const {
        enforce(name in materials, "Material group not found: " ~ name);
        enforce(name in pointCounts, "Point count not found for material group: " ~ name);
        enforce(pointCounts[name] > 0, "Material group has no points: " ~ name);

        // Scale force by number of points in the group
        return materials[name].force / pointCounts[name].to!double;
    }

    /// Update point count for a material group
    void incrementPointCount(string name) {
        if (name !in pointCounts) {
            pointCounts[name] = 0;
        }
        pointCounts[name]++;
    }

    /// Merge properties from multiple material groups in sequence
    MaterialConfig!V mergeGroups(const string[] groupNames) const {
        enforce(groupNames.length > 0, "At least one material group must be specified");
        
        // Start with empty config and merge each group in sequence
        MaterialConfig!V result;
        foreach (name; groupNames) {
            result = result.merge(getMaterial(name));
        }

        // Validate final configuration has required properties
        enforce(result.density > 0.0, 
            "No density defined in material groups: [%(%s, %)]".format(groupNames));
        enforce(result.youngsModulus > 0.0, 
            "No Young's modulus defined in material groups: [%(%s, %)]".format(groupNames));
        enforce(result.criticalStretch > 0.0, 
            "No critical stretch defined in material groups: [%(%s, %)]".format(groupNames));
            
        return result;
    }
}

/// Parse a vector from a JSON array
private V parseVector(V)(JSONValue json) {
    enforce(json.type == JSONType.array, 
        "Vector must be a JSON array, got %s".format(json.type));
    
    auto arr = json.array;
    enforce(arr.length == V.length, 
        "Vector dimension mismatch: expected %d components, got %d"
        .format(V.length, arr.length));
    
    static if (V.length == 1) {
        return V(arr[0].get!double);
    } else static if (V.length == 2) {
        return V(arr[0].get!double, arr[1].get!double);
    } else static if (V.length == 3) {
        return V(arr[0].get!double, arr[1].get!double, arr[2].get!double);
    }
}

/// Parse a material configuration from JSON
private MaterialConfig!V parseMaterialConfig(V)(JSONValue json) {
    MaterialConfig!V config;
    
    // Parse physical properties if present
    if ("density" in json) {
        config.density = json["density"].get!double;
        enforce(config.density > 0.0, "Density must be positive");
    }
    
    if ("youngsModulus" in json) {
        config.youngsModulus = json["youngsModulus"].get!double;
        enforce(config.youngsModulus > 0.0, "Young's modulus must be positive");
    }
    
    if ("criticalStretch" in json) {
        config.criticalStretch = json["criticalStretch"].get!double;
        enforce(config.criticalStretch > 0.0, "Critical stretch must be positive");
    }
    
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

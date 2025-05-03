module main;

import std.stdio;
import std.getopt;
import std.exception;
import std.string;
import std.path;
import std.conv;
import std.file;
import std.sumtype;

import types;
import gmsh;
import writer;

void validatePoints(Point2D[] points) {
    enforce(points.length > 0, "No points generated from mesh");
    
    // Check all volumes are positive
    foreach(point; points) {
        enforce(point.volume > 0.0, 
            "Invalid point volume detected: " ~ point.volume.to!string);
    }
    
    // Check all points have at least one material group
    foreach(point; points) {
        enforce(point.materialGroups.length > 0, 
            "Point found without any material groups");
    }
}

void validatePoints(Point3D[] points) {
    enforce(points.length > 0, "No points generated from mesh");
    
    // Check all volumes are positive
    foreach(point; points) {
        enforce(point.volume > 0.0, 
            "Invalid point volume detected: " ~ point.volume.to!string);
    }
    
    // Check all points have at least one material group
    foreach(point; points) {
        enforce(point.materialGroups.length > 0, 
            "Point found without any material groups");
    }
}

int main(string[] args) {
    Options options;
    
    try {
        auto helpInfo = getopt(
            args,
            "input|i", "Input GMSH mesh file (.msh)", &options.inputFile,
            "output|o", "Output points file (.jsonl)", &options.outputFile,
            "validate|v", "Validate output points", &options.validate
        );

        if (helpInfo.helpWanted || args.length < 2) {
            writeln("gmsh2pd - Convert GMSH mesh to peridynamics points");
            writeln();
            defaultGetoptPrinter("Usage: gmsh2pd [options]\n", helpInfo.options);
            return 0;
        }

        enforce(!options.inputFile.empty, "Input file path required");
        enforce(!options.outputFile.empty, "Output file path required");
        enforce(options.inputFile.exists, "Input file does not exist: " ~ options.inputFile);
        enforce(options.inputFile.extension == ".msh", "Input file must be a .msh file");
        
        // Initialize GMSH
        int err;
        gmshInitialize(cast(int)args.length, cast(const(char)**)args.ptr, 0, 0, &err);
        enforce(err == 0, "Failed to initialize GMSH");
        scope(exit) gmshFinalize(&err);
        
        // Load mesh file
        gmshOpen(options.inputFile.toStringz, &err);
        enforce(err == 0, "Failed to open mesh file: " ~ options.inputFile);
        
        // Get mesh dimension
        int dim;
        gmshModelGetDimension(&dim, &err);
        enforce(err == 0, "Failed to get mesh dimension");
        enforce(dim == 2 || dim == 3, "Only 2D and 3D meshes are supported");
        
        // Process mesh
        auto converter = new GmshConverter();
        auto points = converter.convert(options);
        
        // Create writer
        auto writer = new PointWriter();

        // Handle points based on dimension
        points.match!(
            (Point2D[] pts) {
                if (options.validate) validatePoints(pts);
                writer.write(options.outputFile, pts);
                writefln("Successfully converted %d 2D points to %s", pts.length, options.outputFile);
            },
            (Point3D[] pts) {
                if (options.validate) validatePoints(pts);
                writer.write(options.outputFile, pts);
                writefln("Successfully converted %d 3D points to %s", pts.length, options.outputFile);
            }
        );
        
        return 0;
        
    } catch (Exception e) {
        stderr.writefln("Error: %s", e.msg);
        return 1;
    }
}

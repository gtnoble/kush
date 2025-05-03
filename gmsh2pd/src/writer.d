module writer;

import std.stdio;
import std.json;
import std.array;
import std.algorithm;
import std.exception;
import std.range : array;

import types: Point2D, Point3D;

/**
 * Handles writing point data to JSONL format
 */
class PointWriter {
    private:
        File outFile;

    public:
        /**
         * Write 2D points to JSONL file
         * Params:
         *   filename = Output file path
         *   points = Array of 2D points to write
         * Throws: FileException if file cannot be opened or written
         */
        void write(string filename, Point2D[] points) {
            try {
                outFile = File(filename, "w");
            } catch (Exception e) {
                throw new Exception("Failed to open output file: " ~ e.msg);
            }
            scope(exit) outFile.close();

            foreach(point; points) {
                auto json = JSONValue([
                    "position": JSONValue(point.position[].map!(x => JSONValue(x)).array),
                    "volume": JSONValue(point.volume),
                    "material_groups": JSONValue(point.materialGroups.map!(x => JSONValue(x)).array)
                ]);
                outFile.writeln(json.toString());
            }
        }

        /**
         * Write 3D points to JSONL file
         * Params:
         *   filename = Output file path
         *   points = Array of 3D points to write
         * Throws: FileException if file cannot be opened or written
         */
        void write(string filename, Point3D[] points) {
            try {
                outFile = File(filename, "w");
            } catch (Exception e) {
                throw new Exception("Failed to open output file: " ~ e.msg);
            }
            scope(exit) outFile.close();

            foreach(point; points) {
                auto json = JSONValue([
                    "position": JSONValue(point.position[].map!(x => JSONValue(x)).array),
                    "volume": JSONValue(point.volume),
                    "material_groups": JSONValue(point.materialGroups.map!(x => JSONValue(x)).array)
                ]);
                outFile.writeln(json.toString());
            }
        }
}

module types;

/**
 * Command-line options
 */
struct Options {
    string inputFile;
    string outputFile;
    bool validate = false;
}

/**
 * 2D point representation
 */
struct Point2D {
    double[2] position;
    double volume;
    string[] materialGroups;
}

/**
 * 3D point representation
 */
struct Point3D {
    double[3] position;
    double volume;
    string[] materialGroups;
}

module gmsh;

import std.exception;
import std.string;
import std.math;
import std.algorithm;
import std.array;
import std.format;
import std.conv;
import std.sumtype;  // For sum type return

import types;

// GMSH element type constants
enum {
    GMSH_TRI3 = 2,  // 3-node triangle
    GMSH_TET4 = 4   // 4-node tetrahedron
}

extern(C) {
    void gmshInitialize(const int argc, const char** argv, const int readConfigFiles, const int run, int* ierr);
    void gmshFinalize(int* ierr);
    void gmshOpen(const char* fileName, int* ierr);
    void gmshModelGetDimension(int* dim, int* ierr);
    void gmshModelGetPhysicalGroups(int** dimTags, size_t* dimTags_n, const int dim, int* ierr);
    void gmshModelGetPhysicalName(const int dim, const int tag, char** name, int* ierr);
    void gmshModelMeshGetElements(int** elementTypes, size_t* elementTypes_n,
                                 size_t*** elementTags, size_t** elementTags_n, size_t* elementTags_nn,
                                 size_t*** nodeTags, size_t** nodeTags_n, size_t* nodeTags_nn,
                                 const int dim, const int tag, int* ierr);
    void gmshModelMeshGetNodes(size_t** nodeTags, size_t* nodeTags_n,
                              double** coord, size_t* coord_n,
                              double** parametricCoord, size_t* parametricCoord_n,
                              const int dim, const int tag,
                              const int includeBoundary,
                              const int returnParametricCoord,
                              int* ierr);
    void gmshModelGetPhysicalGroupsForEntity(const int dim, const int tag, int** physicalTags, size_t* physicalTags_n, int* ierr);
}

private string elementTypeName(int type) {
    switch(type) {
        case GMSH_TRI3: return "triangle";
        case GMSH_TET4: return "tetrahedron";
        default: return format("type-%d", type);
    }
}

struct Node {
    size_t tag;
    double[] coord;
}

alias Points = SumType!(Point2D[], Point3D[]);

class GmshConverter {
    private:
        int dimension;
        int[string] physicalGroupTags;
        Node[size_t] nodes;
        
    private:
        double calculateTriangleVolume(size_t[] nodeTags) {
            auto n1 = nodes[nodeTags[0]].coord;
            auto n2 = nodes[nodeTags[1]].coord;
            auto n3 = nodes[nodeTags[2]].coord;

            return abs((n2[0] - n1[0]) * (n3[1] - n1[1]) - 
                      (n3[0] - n1[0]) * (n2[1] - n1[1])) / 2.0;
        }

        double calculateTetraVolume(size_t[] nodeTags) {
            auto n1 = nodes[nodeTags[0]].coord;
            auto n2 = nodes[nodeTags[1]].coord;
            auto n3 = nodes[nodeTags[2]].coord;
            auto n4 = nodes[nodeTags[3]].coord;

            auto v1 = [n2[0] - n1[0], n2[1] - n1[1], n2[2] - n1[2]];
            auto v2 = [n3[0] - n1[0], n3[1] - n1[1], n3[2] - n1[2]];
            auto v3 = [n4[0] - n1[0], n4[1] - n1[1], n4[2] - n1[2]];

            return abs(
                v1[0] * (v2[1]*v3[2] - v2[2]*v3[1]) -
                v1[1] * (v2[0]*v3[2] - v2[2]*v3[0]) +
                v1[2] * (v2[0]*v3[1] - v2[1]*v3[0])
            ) / 6.0;
        }

        double[] calculateCentroid(size_t[] nodeTags) {
            double[] centroid;
            if (dimension == 2) {
                centroid = [0.0, 0.0];
            } else {
                centroid = [0.0, 0.0, 0.0];
            }

            foreach(tag; nodeTags) {
                auto nodeCoord = nodes[tag].coord;
                foreach(i; 0..dimension) {
                    centroid[i] += nodeCoord[i];
                }
            }

            foreach(ref c; centroid) {
                c /= nodeTags.length;
            }

            return centroid;
        }

        string[] getElementGroups(int dim, int tag) {
            int err;
            int* physicalTags;
            size_t numPhysicalTags;

            gmshModelGetPhysicalGroupsForEntity(dim, tag, &physicalTags, &numPhysicalTags, &err);
            enforce(err == 0, "Failed to get physical groups for element");

            string[] groups;
            for (size_t i = 0; i < numPhysicalTags; i++) {
                char* namePtr;
                gmshModelGetPhysicalName(dim, physicalTags[i], &namePtr, &err);
                enforce(err == 0, "Failed to get physical group name");
                groups ~= namePtr.fromStringz.idup;
            }

            return groups;
        }

        void loadPhysicalGroups() {
            int err;
            int* dimTags;
            size_t numDimTags;

            gmshModelGetPhysicalGroups(&dimTags, &numDimTags, -1, &err);
            enforce(err == 0, "Failed to get physical groups");

            for (size_t i = 0; i < numDimTags; i += 2) {
                int dim = dimTags[i];
                int tag = dimTags[i + 1];
                
                char* namePtr;
                gmshModelGetPhysicalName(dim, tag, &namePtr, &err);
                enforce(err == 0, "Failed to get physical group name");
                
                string name = namePtr.fromStringz.idup;
                physicalGroupTags[name] = tag;
            }
        }

        void loadNodes() {
            int err;
            size_t* nodeTags;
            size_t numNodeTags;
            double* coords;
            size_t numCoords;
            double* paramCoords;
            size_t numParamCoords;

            gmshModelMeshGetNodes(&nodeTags, &numNodeTags,
                                &coords, &numCoords,
                                &paramCoords, &numParamCoords,
                                -1, -1, 1, 0, &err);
            enforce(err == 0, "Failed to get nodes");

            for (size_t i = 0; i < numNodeTags; i++) {
                double[] nodeCoords;
                if (dimension == 2) {
                    nodeCoords = [coords[i*3], coords[i*3 + 1]];
                } else {
                    nodeCoords = [coords[i*3], coords[i*3 + 1], coords[i*3 + 2]];
                }
                nodes[nodeTags[i]] = Node(nodeTags[i], nodeCoords);
            }
        }

        Point2D[] convert2D() {
            int err;
            Point2D[] points;
            
            int* elementTypes;
            size_t numElementTypes;
            size_t** elementTags;
            size_t* elementTags_n;
            size_t elementTags_nn;
            size_t** nodeTags;
            size_t* nodeTags_n;
            size_t nodeTags_nn;

            gmshModelMeshGetElements(&elementTypes, &numElementTypes,
                                   &elementTags, &elementTags_n, &elementTags_nn,
                                   &nodeTags, &nodeTags_n, &nodeTags_nn,
                                   2, -1, &err);
            enforce(err == 0, "Failed to get 2D elements");

            for (size_t i = 0; i < numElementTypes; i++) {
                enforce(elementTypes[i] == GMSH_TRI3,
                    format("Only triangular elements are supported, found %s", 
                           elementTypeName(elementTypes[i])));

                for (size_t j = 0; j < elementTags_n[i]; j++) {
                    auto elemTag = elementTags[i][j];
                    auto elemNodes = nodeTags[i][j*3..(j+1)*3];
                    
                    auto volume = calculateTriangleVolume(elemNodes);
                    auto centroid = calculateCentroid(elemNodes);
                    auto groups = getElementGroups(2, cast(int)elemTag);
                    
                    points ~= Point2D([centroid[0], centroid[1]], volume, groups);
                }
            }

            return points;
        }

        Point3D[] convert3D() {
            int err;
            Point3D[] points;
            
            int* elementTypes;
            size_t numElementTypes;
            size_t** elementTags;
            size_t* elementTags_n;
            size_t elementTags_nn;
            size_t** nodeTags;
            size_t* nodeTags_n;
            size_t nodeTags_nn;

            gmshModelMeshGetElements(&elementTypes, &numElementTypes,
                                   &elementTags, &elementTags_n, &elementTags_nn,
                                   &nodeTags, &nodeTags_n, &nodeTags_nn,
                                   3, -1, &err);
            enforce(err == 0, "Failed to get 3D elements");

            for (size_t i = 0; i < numElementTypes; i++) {
                enforce(elementTypes[i] == GMSH_TET4,
                    format("Only tetrahedral elements are supported, found %s", 
                           elementTypeName(elementTypes[i])));

                for (size_t j = 0; j < elementTags_n[i]; j++) {
                    auto elemTag = elementTags[i][j];
                    auto elemNodes = nodeTags[i][j*4..(j+1)*4];
                    
                    auto volume = calculateTetraVolume(elemNodes);
                    auto centroid = calculateCentroid(elemNodes);
                    auto groups = getElementGroups(3, cast(int)elemTag);
                    
                    points ~= Point3D([centroid[0], centroid[1], centroid[2]], 
                                    volume, groups);
                }
            }

            return points;
        }

    public:
        Points convert(Options options) {
            int err;
            gmshModelGetDimension(&dimension, &err);
            enforce(err == 0, "Failed to get mesh dimension");

            loadPhysicalGroups();
            loadNodes();

            if (dimension == 2) {
                return Points(convert2D());
            } else if (dimension == 3) {
                return Points(convert3D());
            } else {
                throw new Exception("Unsupported mesh dimension");
            }
        }
}

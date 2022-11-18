#pragma once
/**
 * @file: include/testbed/common.h
 * @author: sailing-innocent
 * @create: 2022-10-15
 * @desp: The Common Definitions for Testbed
*/

#ifndef TESTBED_COMMON_H_
#define TESTBED_COMMON_H_

#define TESTBED_NAMESPACE_BEGIN namespace testbed {
#define TESTBED_NAMESPACE_END }

TESTBED_NAMESPACE_BEGIN

// using Eigen Matrix

enum class IMeshRenderMode : int {
    Off,
    VertexColors,
    // VertexNormals,
    // FaceIDs,
};

enum class IGroundTruthRenderMode : int {
    Shade,
    Depth,
    NumRenderModes,
};

static constexpr const char* GroundTruthRenderModeStr = "Shade\0Depoth\0\0";

enum class IRenderMode : int {
    AO,
    Shade,
    Normals,
    Positions,
    Depth,
    // Distortion,
    // Cost,
    // Slice
    NumRenderModes,
    EncodingVis, // Encoding Vis exists outside of the standard render modes
};

static constexpr const char* RenderModeStr = "AO\0Shade\0Normals\0Positions\0Depth\0\0";

enum class IRandomMode : int {
    Random,
    Halton,
    // Sobol,
    // Stratified.
    NumImageRandomModes,
};

static constexpr const char* RandomModeStr = "Random\0Halton\0\0";

// LossType
// LossTypeStr
// Nerf Activation
// MeshSdfMode
// ColorSpace
// TonemapCurve
// DLSS Quality
// 
enum ITestBedMode {
    RaytraceMesh
    // SpheretraceMesh,
    // SDFBricks,
};

struct Ray {
    // o
    // d
};

// Training X Form
// ElensMode
// Lens

// ----------------------- UTILITY FUNCTIONS ---------------------------
// sign()
// binary_search

// --------------- END OF UTILITY FUNCTIONS ----------------------------

// Timer

TESTBED_NAMESPACE_END

#endif // TESTBED_COMMON_H_
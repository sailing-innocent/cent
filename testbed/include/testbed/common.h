#pragma once
/**
 * @file: include/testbed/common.h
 * @author: sailing-innocent
 * @create: 2022-10-15
 * @desp: The Common Definitions for Testbed
*/

#ifndef TESTBED_COMMON_H_
#define TESTBED_COMMON_H_

#include <Eigen/Dense>
#define TESTBED_NAMESPACE_BEGIN namespace testbed {
#define TESTBED_NAMESPACE_END }


TESTBED_NAMESPACE_BEGIN

using Vector2i32 = Eigen::Matrix<uint32_t, 2, 1>;
using Vector3i32 = Eigen::Matrix<uint32_t, 3, 1>;
using Vector4i16 = Eigen::Matrix<uint16_t, 4, 1>;
using Vector4i32 = Eigen::Matrix<uint32_t, 4, 1>;

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
enum ITestbedMode {
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
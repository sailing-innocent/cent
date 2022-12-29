#ifndef CAMERA_H_
#define CAMERA_H_

#include <vector>

enum CAMERA_MOVEMENT {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

// camera default values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2000.0f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 60.0f;
const float MAX_FOV = 100.0f;

class Camera
{

};

#endif // CAMERA_H_
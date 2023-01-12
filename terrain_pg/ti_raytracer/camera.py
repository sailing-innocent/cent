import taichi as ti
import taichi.math as tm
import math


def cross3(x, y):
    return tm.vec3(x[1] * y[2] - x[2] * y[1],
                   x[2] * y[0] - x[0] * y[2],
                   x[0] * y[1] - x[1] * y[0])


def norm(x):
    n = len(x)
    xsum = 0.0
    for i in range(n):
        xsum = xsum + x[i] * x[i]
    return x / math.sqrt(xsum)


@ti.data_oriented
class Camera:
    def __init__(self,
                 aspect_ratio=16.0/9.0,
                 viewport_height=2.0,
                 center=tm.vec3(0.0, 0.0, 0.0),
                 lookat=tm.vec3(0.0, 0.0, -1.0)):
        self.ar = aspect_ratio
        self.vh = viewport_height
        self.vw = self.vh * self.ar
        self.origin = center
        self.principle_dir = norm(lookat - center)
        self.up = tm.vec3(0, 1, 0)
        self.horizental = cross3(self.principle_dir, self.up)
        self.vertical = cross3(self.horizental, self.principle_dir)
        self.focal_length = 1.0
        self.horizental = self.vw * norm(self.horizental)
        self.vertical = self.vh * norm(self.vertical)
        print(self.horizental)
        print(self.vertical)
        self.ll_corner = self.origin - self.horizental/2 - \
            self.vertical/2 + self.focal_length * self.principle_dir

    @ti.func
    def get_ray(self, u, v):
        dir = self.ll_corner + u * self.horizental + v * self.vertical - self.origin
        return self.origin, dir

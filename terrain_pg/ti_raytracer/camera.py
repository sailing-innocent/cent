import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Camera:
    def __init__(self,
                 aspect_ratio=16.0/9.0,
                 viewport_height=2.0):
        self.ar = aspect_ratio
        self.vh = viewport_height
        self.vw = self.vh * self.ar
        self.origin = tm.vec3(0, 0, 0)
        self.focal_length = 1.0
        self.horizental = tm.vec3(self.vw, 0, 0)
        self.vertical = tm.vec3(0, self.vh, 0)
        self.ll_corner = self.origin - self.horizental/2 - \
            self.vertical/2 - tm.vec3(0, 0, self.focal_length)

    @ti.func
    def get_ray(self, u, v):
        dir = self.ll_corner + u * self.horizental + v * self.vertical - self.origin
        return self.origin, dir

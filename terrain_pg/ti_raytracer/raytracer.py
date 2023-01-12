
import taichi as ti
import taichi.math as tm
from .ray import Rays
from .camera import Camera
from .world import Triangle, Sphere, World

ti.init(arch=ti.gpu)


@ti.func
def ray_color(hittable_world, ray_org, ray_dir):
    color = tm.vec3(1, 0, 0)
    hit, p, N, front_facing, id = hittable_world.hit_all(ray_org, ray_dir)
    if (hit):
        color = 0.5 * (N + tm.vec3(1, 1, 1))
    else:
        unit_dir = tm.normalize(ray_dir)
        t = 0.5 * (unit_dir.y + 1.0)
        color = (1.0 - t) * tm.vec3(1.0, 1.0, 1.0) + t * tm.vec3(0.5, 0.7, 1.0)
    return color


@ti.data_oriented
class tracer:
    def __init__(self, _world=World(), _camera=Camera()):
        self.aspect_ratio = 16.0 / 9.0
        self.image_width = 1600
        self.image_height = (int)(self.image_width/self.aspect_ratio)
        self.pixels = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.image_width, self.image_height))
        self.camera = _camera
        self.rays = Rays(self.image_width, self.image_height)
        self.world = _world
        self.world.commit()

    @ti.func
    def write_color(self, i, j, color):
        self.pixels[i, j] = color

    @ti.kernel
    def render(self):
        for i, j in self.pixels:
            u = i / (self.image_width - 1)
            v = j / (self.image_height - 1)
            ray_org, ray_dir = self.camera.get_ray(u, v)
            self.rays.set(i, j, ray_org, ray_dir)
            color = ray_color(self.world, ray_org, ray_dir)
            self.write_color(i, j, color)

    def save(self, figpath="result.jpg"):
        ti.tools.imwrite(self.pixels.to_numpy(), figpath)

import taichi as ti
import taichi.math as tm

from ti_raytracer.raytracer import tracer
from ti_raytracer.world import World
from ti_raytracer.camera import Camera
from scenes.block import Block
from scenes.terrain import Terrain

import math


def sample_func(x, y):
    return 1 * math.sin(0.25 * (x*x+y*y))


def debug():
    myworld = World()
    camera_center = tm.vec3(0.0, 0.0, 12.0)
    lookat = tm.vec3(0.0, 0.0, -1.0)
    mycamera = Camera(center=camera_center, lookat=lookat)
    terrain = Terrain(64, 64, 0.5)
    terrain.sample(sample_func)
    mesh = terrain.gen_mesh()
    for tri in mesh.triangles:
        myworld.add_triangle(tri)
    tr = tracer(myworld, mycamera)
    tr.render()
    tr.save()


if __name__ == "__main__":
    debug()

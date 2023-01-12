import taichi as ti
import taichi.math as tm

from ti_raytracer.raytracer import tracer
from ti_raytracer.world import World
from ti_raytracer.camera import Camera
from scenes.block import Block
from scenes.terrain import Terrain

import math
import time 

import numpy as np


def sample_func(x, y):
    return 1 * math.sin(0.25 * math.sqrt(x*x+y*y))


def debug():
    """
    myworld = World()
    camera_center = tm.vec3(0.0, 0.0, 2.0)
    lookat = tm.vec3(0.0, 0.0, -1.0)
    mycamera = Camera(center=camera_center, lookat=lookat)
    terrain = Terrain(4, 4, 0.5)
    terrain.sample(sample_func)
    st = time.time()
    # mesh = terrain.gen_mesh()
    terrain.gen_mesh_ti()
    print("it requires: ", time.time() - st, "  to generate mesh")
    print(terrain.mesh)
    """
    @ti.kernel
    def paint():
        print("hhh")

    paint()

    """
    st = time.time()
    for tri in mesh.triangles:
        myworld.add_triangle(tri)
    tr = tracer(myworld, mycamera)
    print("it requires: ", time.time() - st, " to load mesh")
    st = time.time()
    tr.render()
    print("it requires: ", time.time() - st, " to render")
    tr.save()
    """

if __name__ == "__main__":
    debug()

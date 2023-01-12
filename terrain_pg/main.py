import taichi as ti
import taichi.math as tm

from ti_raytracer.raytracer import tracer
import ti_raytracer.world as world
from ti_raytracer.camera import Camera
from scenes.block import Block


def debug():
    myworld = world.World()
    # mycamera = Camera()
    block = Block(-0.5, -0.5, -1.5, 1, 1, -1)
    mesh = block.gen_mesh()
    for tri in mesh.triangles:
        myworld.add_triangle(tri)
    tr = tracer(myworld)
    tr.render()  # tr.render(world, camera)
    tr.save()


if __name__ == "__main__":
    debug()

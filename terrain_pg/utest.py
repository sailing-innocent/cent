import taichi as ti
import taichi.math as tm

from ti_raytracer.raytracer import tracer
import ti_raytracer.world as world
from scenes.block import Block
from scenes.terrain import Terrain
ti.init(arch=ti.gpu)


def triangles_test():
    myworld = world.World()
    # myworld.add_sphere(world.Sphere(tm.vec3(0, 0, -1), 0.5))

    p1 = tm.vec3(-0.5, -0.5, -1.0)
    p2 = tm.vec3(-0.5, 0.5, -1.0)
    p3 = tm.vec3(0.5, -0.5, -1.0)
    p4 = tm.vec3(0.5, 0.5, -1.0)
    tr1 = world.Triangle(p1, p3, p2)
    tr2 = world.Triangle(p4, p2, p3)

    p5 = tm.vec3(-0.5, -0.5, -2.0)
    p6 = tm.vec3(-0.5, 0.5, -2.0)
    p7 = tm.vec3(0.5, -0.5, -2.0)
    tr3 = world.Triangle(p5, p6, p7)

    # myworld.add_triangle(tr1)
    myworld.add_triangle(tr2)
    myworld.add_triangle(tr3)
    """
    block = Block(-0.5, -0.5, -1.5, 1, 1, -1)
    mesh = block.gen_mesh()
    for tri in mesh.triangles:
        myworld.add_triangle(tri)
    """
    tr = tracer(myworld)
    tr.render()  # tr.render(world, camera)
    tr.save()


def block_test():
    myworld = world.World()
    block = Block(-0.5, -0.5, -1.5, 1, 1, 1)
    mesh = block.gen_mesh()
    for tri in mesh.triangles:
        myworld.add_triangle(tri)
    tr = tracer(myworld)
    tr.render()
    tr.save()


def sample_func(x, y):
    return -(x*x+y*y)+1


def terrain_test():
    terrain = Terrain()
    terrain.sample(sample_func)


if __name__ == "__main__":
    # triangles_test()
    terrain_test()

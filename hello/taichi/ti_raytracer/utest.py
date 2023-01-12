import taichi as ti
import taichi.math as tm

from raytracer import tracer
import world


def debug():
    myworld = world.World()
    # myworld.add_sphere(world.Sphere(tm.vec3(0, 0, -1), 0.5))
    myworld.add_triangle(world.Triangle())
    tr = tracer(myworld)
    tr.render()  # tr.render(world, camera)
    tr.save()


if __name__ == "__main__":
    debug()

import taichi as ti
import taichi.math as tm
import ray


@ti.func
def is_front_facing(ray_dir, out_normal):
    return tm.dot(ray_dir, out_normal) < 0.0


class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.id = -1

@ti.func
def hit_sphere(center, radius, ray_org, ray_dir, t_min, t_max):
    oc = ray_org - center
    a = tm.dot(ray_dir, ray_dir)
    half_b = tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - radius * radius
    discriminate = half_b * half_b - a * c

    hit = discriminate > 0
    root = -1.0
    if (hit):
        sqrtd = tm.sqrt(discriminate)

        root = (-half_b - sqrtd) / a
        if (root < t_min or root > t_max):
            root = (-half_b + sqrtd) / a
            if (root < t_min or root > t_max):
                hit = False
    return hit, root


class Triangle:
    def __init__(self, a, b, c, order):
        self.points = [a,b,c]
        self.facing = 1 # abc or -1 cab for normal calculation

@ti.func
def hit_triangle(center, a, b, c, ray_org, ray_dir, t_min, t_max):
    pass # https://zhuanlan.zhihu.com/p/73686686


@ti.data_oriented
class World:
    def __init__(self):
        self.spheres = []

    def add_sphere(self, sphere):
        sphere.id = len(self.spheres)
        self.spheres.append(sphere)

    def commit(self):
        self.n = len(self.spheres)
        self.radius = ti.field(ti.f32, shape=(self.n))
        self.center = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.n))

        for i in range(self.n):
            self.radius[i] = self.spheres[i].radius
            self.center[i] = self.spheres[i].center
        del self.spheres

    @ti.func
    def hit_all(self, ray_org, ray_dir):
        hit_anything = False
        t_min = 0.00001
        closest_so_far = 999999999.9
        hit_index = -1

        # init p and n
        p = tm.vec3(0, 0, 0)
        n = tm.vec3(1, 0, 0)
        front_facing = True
        # TODO bvh tree
        for i in range(self.n):
            hit, t = hit_sphere(
                self.center[i], self.radius[i], ray_org, ray_dir, t_min, closest_so_far)
            if hit:
                if (closest_so_far > t):
                    hit_anything = True
                    closest_so_far = t
                    hit_index = i

        if hit_anything:
            p = ray.at(ray_org, ray_dir, closest_so_far)
            n = (p - self.center[hit_index]) / self.radius[hit_index]
            front_facing = is_front_facing(ray_dir, n)
            if not front_facing:
                n = -n

        return hit_anything, p, n, front_facing, hit_index

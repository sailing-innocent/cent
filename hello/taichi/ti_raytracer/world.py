import taichi as ti
import taichi.math as tm


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

        root = (-half_b-sqrtd) / a
        if (root < t_min or root > t_max):
            root = (-half_b+sqrtd) / a
            if (root < t_min or root > t_max):
                hit = False
    return hit, root


@ti.data_oriented
class World:
    def __init__(self):
        self.spheres = []

    def add_sphere(self, sphere):
        sphere.id = len(self.spheres)
        self.spheres.append(sphere)

    @ti.func
    def hit_all(self, ray_org, ray_dir):
        hit_anything = False
        t_min = 0.00001
        closest_so_far = 999999999.9
        hit_index = 0

        # TODO bvh tree
        test_sphere = self.spheres[0]
        hit, root = hit_sphere(test_sphere)

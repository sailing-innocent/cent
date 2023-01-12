import taichi as ti
import taichi.math as tm
from .ray import at


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
    n = tm.vec3(0, 0, 0)
    if (hit):
        sqrtd = tm.sqrt(discriminate)

        root = (-half_b - sqrtd) / a
        if (root < t_min or root > t_max):
            root = (-half_b + sqrtd) / a
            if (root < t_min or root > t_max):
                hit = False

        p = at(ray_org, ray_dir, root)
        n = (p - center) / radius
    return hit, root, n


class Triangle:
    def __init__(self,
                 pa=tm.vec3(0.5, 0.5, -1),
                 pb=tm.vec3(-0.5, 0.5, -1),
                 pc=tm.vec3(0, -0.5, -1)):
        # the triangle normal is defined by papb \cross papc
        self.pa = pa
        self.pb = pb
        self.pc = pc
        self.id = -1

    def __str__(self):
        return "Triangle {}: {}, {}, {}".format(self.id, self.pa, self.pb, self.pc)


@ti.func
def hit_triangle(pa, pb, pc, ray_org, ray_dir, t_min, t_max):
    # print("hitting")
    # print(pa, pb, pc, ray_org, ray_dir)
    hit = True
    ab = pb - pa
    ac = pc - pa
    normal = tm.cross(ab, ac)
    oa = pa - ray_org
    dir_dot_normal = tm.dot(ray_dir, normal)
    p = tm.vec3(0, 0, 0)
    t = 0.0
    if (dir_dot_normal < 0.01 and dir_dot_normal > -0.01):
        hit = False
    if (hit):
        t = tm.dot(oa, normal)/dir_dot_normal
        if (t < t_min or t > t_max):
            hit = False
        p = at(ray_org, ray_dir, t)
        ap = p - pa
        det1 = tm.dot(tm.cross(ab, ap), tm.cross(ap, ac))
        ba = -ab
        bc = pc - pb
        bp = p - pb
        det2 = tm.dot(tm.cross(ba, bp), tm.cross(bp, bc))
        ca = -ac
        cb = -bc
        cp = p - pc
        det3 = tm.dot(tm.cross(ca, cp), tm.cross(cp, cb))
        hit = det1 > 0 and det2 > 0 and det3 > 0

    return hit, t, normal


class Mesh:
    def __init__(self):
        self.triangles = []

    def add_triangle(self, triangle):
        triangle.id = len(self.triangles)
        self.triangles.append(triangle)

    def append_mesh(self, mesh):
        for tri in mesh.triangles:
            self.add_triangle(tri)

    def __str__(self):
        str = ""
        for tri in self.triangles:
            str = str + "\n" + tri.__str__()
        return str


@ti.data_oriented
class World:
    def __init__(self):
        self.spheres = []
        self.triangles = []

    def local_T_world(self, p):
        # local xyz and world render xzy
        return tm.vec3(p[0], p[1], p[2])

    def add_triangle(self, triangle):
        triangle.id = len(self.triangles)
        self.triangles.append(triangle)

    def add_sphere(self, sphere):
        sphere.id = len(self.spheres)
        self.spheres.append(sphere)

    def commit(self):
        """
        # COMMIT SPHERES
        self.n = len(self.spheres)
        if (self.n <= 0):
            print("WARNING: the world is empty")
        self.radius = ti.field(ti.f32, shape=(self.n))
        self.center = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.n))
        for i in range(self.n):
            self.radius[i] = self.spheres[i].radius
            self.center[i] = self.spheres[i].center
        del self.spheres
        """

        # COMMIT TRIANGLES
        self.tn = len(self.triangles)
        self.tri = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.tn, 3))
        for i in range(self.tn):
            # print(self.triangles[i])
            self.tri[i, 0] = self.local_T_world(self.triangles[i].pa)
            self.tri[i, 1] = self.local_T_world(self.triangles[i].pb)
            self.tri[i, 2] = self.local_T_world(self.triangles[i].pc)

    @ti.func
    def hit_all(self, ray_org, ray_dir):
        hit_anything = False
        t_min = 0.00001
        closest_so_far = 999999999.9
        hit_index = -1

        p = tm.vec3(0, 0, 0)
        hit_normal = tm.vec3(1, 0, 0)
        front_facing = True
        # TODO bvh tree
        """
        for i in range(self.n):
            hit, t, n = hit_sphere(
                self.center[i], self.radius[i], ray_org, ray_dir, t_min, closest_so_far)
            if hit:
                if (closest_so_far > t):
                    hit_anything = True
                    closest_so_far = t
                    hit_index = i
        """
        hit_any_triangle = False
        for i in range(self.tn):
            hit, t, n = hit_triangle(
                self.tri[i, 0], self.tri[i, 1], self.tri[i, 2], ray_org, ray_dir, t_min, closest_so_far)

            if hit:
                # print(ray_dir)
                # print(n)
                if (closest_so_far > t):
                    hit_any_triangle = True
                    closest_so_far = t
                    hit_index = i
                    hit_normal = n

        if hit_any_triangle:
            hit_anything = True
            # print(ray_dir)

        if hit_anything:
            front_facing = is_front_facing(ray_dir, hit_normal)
            if not front_facing:
                # n = -n
                hit_normal = hit_normal

        return hit_anything, p, hit_normal, front_facing, hit_index

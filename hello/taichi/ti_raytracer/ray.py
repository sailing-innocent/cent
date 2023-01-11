import taichi as ti


@ti.func
def at(origin, dir, t):
    return origin + t * dir


@ti.data_oriented
class Rays:
    def __init__(self, x, y):
        self.origin = ti.Vector.field(n=3, dtype=ti.f32, shape=(x, y))
        self.direction = ti.Vector.field(n=3, dtype=ti.f32, shape=(x, y))

    @ti.func
    def set(self, x, y, ray_org, ray_dir):
        self.origin[x, y] = ray_org
        self.direction[x, y] = ray_dir

    @ti.func
    def get_od(self, x, y):
        return self.origin[x, y], self.direction[x, y]


@ti.data_oriented
class HitRecord:
    def __init__(self, x, y):
        self.hit = ti.field(ti.i32, shape=(x, y))
        self.point = ti.Vector.field(n=3, dtype=ti.f32, shape=(x, y))
        self.normal = ti.Vector.field(n=3, dtype=ti.f32, shape=(x, y))
        self.front_facing = ti.field(ti.i32, shape=(x, y))

    @ti.func
    def set(self, x, y, hit, point, normal, front_facing):
        self.hit[x, y] = hit
        self.point = point
        self.normal = normal
        self.front_facing = front_facing

    @ti.func
    def get(self, x, y):
        return self.hit[x, y]

    @ti.func
    def get_hit(self, x, y):
        return self.hit[x, y]

    @ti.func
    def set_hit(self, x, y, hit):
        self.hit[x, y] = hit

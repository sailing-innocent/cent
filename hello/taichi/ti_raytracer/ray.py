import taichi as ti


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

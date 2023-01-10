import taichi as ti
import taichi.math as tm
import ray

ti.init(arch=ti.gpu)


@ti.func
def at(t, origin, dir):
    return origin + t * dir


@ti.func
def hit_sphere(center, radius, ray_org, ray_dir):
    oc = ray_org - center
    a = tm.dot(ray_dir, ray_dir)
    half_b = tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - radius * radius
    discriminate = half_b * half_b - a * c
    root = -1.0
    if (discriminate > 0):
        root = (-half_b-tm.sqrt(discriminate)) / a
    return root


@ti.func
def ray_color(ray_org, ray_dir):
    color = tm.vec3(1, 0, 0)
    t = hit_sphere(tm.vec3(0, 0, -1), 0.5, ray_org, ray_dir)
    if (t > 0.0):
        N = tm.normalize(at(t, ray_org, ray_dir)-tm.vec3(0, 0, -1))
        color = 0.5 * tm.vec3(N.x + 1, N.y + 1, N.z + 1)
    else:
        unit_dir = tm.normalize(ray_dir)
        t = 0.5 * (unit_dir.y + 1.0)
        color = (1.0 - t) * tm.vec3(1.0, 1.0, 1.0) + t * tm.vec3(0.5, 0.7, 1.0)
    return color


@ti.data_oriented
class tracer:
    def __init__(self):
        self.aspect_ratio = 16.0 / 9.0
        self.image_width = 400
        self.image_height = (int)(self.image_width/self.aspect_ratio)
        self.pixels = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.image_width, self.image_height))

        self.viewport_height = 2.0
        self.viewport_width = self.viewport_height * self.aspect_ratio
        self.focal_length = 1.0
        self.origin = tm.vec3(0, 0, 0)
        self.horizental = tm.vec3(self.viewport_width, 0, 0)
        self.vertical = tm.vec3(0, self.viewport_height, 0)
        self.ll_corner = self.origin - self.horizental/2 - \
            self.vertical/2 - tm.vec3(0, 0, self.focal_length)

        self.rays = ray.Rays(self.image_width, self.image_height)
        # self.hitRecords = HitRecords()

    @ti.func
    def write_color(self, i, j, color):
        self.pixels[i, j] = color

    @ti.kernel
    def render(self):
        for i, j in self.pixels:
            u = i / (self.image_width - 1)
            v = j / (self.image_height - 1)
            self.rays.set(i, j, self.origin, self.ll_corner + u *
                          self.horizental + v * self.vertical - self.origin)
            ray_org, ray_dir = self.rays.get_od(i, j)
            color = ray_color(ray_org, ray_dir)
            self.write_color(i, j, color)

    def save(self, figpath="result.jpg"):
        ti.tools.imwrite(self.pixels.to_numpy(), figpath)


def debug():
    tr = tracer()
    tr.render()
    tr.save()


if __name__ == "__main__":
    debug()

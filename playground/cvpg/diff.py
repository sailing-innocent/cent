import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n, n))


@ti.func
def in_rect(z, a, b):
    return z[0] > a[0] and z[1] > a[1] and z[0] < b[0] and z[1] < b[1]


@ti.kernel
def paint(t: float):
    a = tm.vec2(-0.5, -0.5)
    b = tm.vec2(0.5, 0.5)
    ada = tm.vec2(-0.52, -0.52)
    bdb = tm.vec2(0.48, 0.48)
    for i, j in pixels:
        z = tm.vec2(i/n-0.5, j/n-0.5) * 2  # center coord
        x1 = 1 if in_rect(z, a, b) else 0
        x2 = 1 if in_rect(z, ada, bdb) else 0
        pixels[i, j] = abs(x1 - x2)


gui = ti.GUI("diff test", res=(n, n))

for i in range(100000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()

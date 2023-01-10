import taichi as ti
import taichi.math as tm
import torch
import cv2 as cv

ti.init(arch=ti.gpu)

width = 800
height = 600
pixels = ti.Vector.field(n=3, dtype=float, shape=(width, height))


@ti.func
def lb_T_o(x):
    return (x - 0.5)*2


@ti.kernel
def paint(t: float):
    for i, j in pixels:
        u = i/width
        ou = lb_T_o(u)
        v = j/height
        ov = lb_T_o(v)
        if (ou * ou + ov * ov) < 0.25:
            pixels[i, j] = tm.vec3(0.1, 0.2, 0.5)
        else:
            pixels[i, j] = tm.vec3(0.6, 0.4, 0.2)


if __name__ == "__main__":
    # use taichi to generate image
    # mapping taichi data to torch data
    # use torch to predict depth
    # compare two images
    gui = ti.GUI("Raytracer", res=(width, height))
    paint(0.0)
    gui.set_image(pixels)
    gui.show("result.jpg")

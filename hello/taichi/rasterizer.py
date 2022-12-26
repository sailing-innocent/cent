import taichi as ti
import taichi.math as tm 

ti.init(arch=ti.gpu)

n = 320
pixels = ti.Vector.field(n=3, dtype=float, shape=(n,n))
N = 2
triangles = ti.Vector.field(n=3, dtype=float, shape=N*3)

@ti.kernel
def init():
    triangles[0] = tm.vec3(0.7, 0.7, 2.0)
    triangles[1] = tm.vec3(0.3, 0.7, 2.0)
    triangles[2] = tm.vec3(0.5, 0.2, 2.0)
    triangles[3] = tm.vec3(0.5, 0.2, 2.0)
    triangles[4] = tm.vec3(0.3, 0.1, 2.0)
    triangles[5] = tm.vec3(0.7, 0.1, 2.0)

@ti.func
def proj(point):
    return tm.vec2(point[0],point[1])

@ti.func
def inTriangle(point1, point2, point3, i, j):
    # print(point1, point2, point3)
    p1 = point2-point1
    p2 = point3-point1
    p = tm.vec2(i-point1[0],j-point1[1])
    alpha = (p[1]*p2[0] - p[0]*p2[1])/(p1[1]*p2[0]-p1[0]*p2[1])
    beta = (p[1]*p1[0] - p[0]*p1[1])/(p2[1]*p1[0]-p2[0]*p1[1])

    flag = False
    if alpha > 0 and beta > 0 and alpha + beta < 1:
        flag = True
    return flag 

@ti.kernel
def paint(t: float):
    for i, j in pixels:
        flag = False
        for idx in range(N):   
            point1 = triangles[idx*3+0]
            point2 = triangles[idx*3+1]
            point3 = triangles[idx*3+2]
            point1_proj = proj(point1)
            point2_proj = proj(point2)
            point3_proj = proj(point3)
            flag = flag or inTriangle(point1_proj, point2_proj, point3_proj, i/n, j/n)
        if flag:
            pixels[i,j] = tm.vec3(0.2, 0.3, 0.6)
        else:
            pixels[i,j] = tm.vec3(0.6, 0.3, 0.2)



init()
gui = ti.GUI("Rasterizer", res = (n, n))
paint(0.0)
gui.set_image(pixels)
gui.show("result.jpg")

# ti.tools.imwrite(pixels.to_numpy(), filename)

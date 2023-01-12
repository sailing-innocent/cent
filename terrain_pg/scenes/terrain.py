import taichi as ti
import taichi.math as tm
import numpy as np
from ti_raytracer.world import Triangle, Mesh
import time

@ti.data_oriented
class Terrain:
    def __init__(self, ixmax=4, iymax=4, usize=0.2):
        self.ixmax = ixmax
        self.iymax = iymax
        self.usize = usize
        self.height_map = ti.field(dtype=ti.f32,shape=(ixmax, iymax))
        self.mesh = ti.Vector.field(n=3, dtype=ti.f32, shape=(ixmax-1, iymax-1, 6))

    def sample(self, sample_func):
        for ix in range(self.ixmax):
            for iy in range(self.iymax):
                x = (ix - (self.ixmax/2))*self.usize
                y = (iy - (self.iymax/2))*self.usize
                self.height_map[ix,iy] = sample_func(x, y)

    @ti.func
    def gen_point(self, ix, iy):
        x = (ix - (self.ixmax/2))*self.usize
        y = (iy - (self.iymax/2))*self.usize
        return tm.vec3(x, y, self.height_map[ix,iy])

    @ti.kernel
    def gen_mesh_ti(self):
        for ix, iy, idx in self.mesh:
            lb = self.gen_point(ix, iy)
            lt = self.gen_point(ix, iy+1)
            rt = self.gen_point(ix+1, iy+1)
            rb = self.gen_point(ix+1, iy)
            if idx == 0:
                self.mesh[ix, iy, idx] = lb
            elif idx == 1:
                self.mesh[ix, iy, idx] = rb
            elif idx == 2:
                self.mesh[ix, iy, idx] = lt
            elif idx == 3:
                self.mesh[ix, iy, idx] = rt
            elif idx == 4:
                self.mesh[ix, iy, idx] = lt
            elif idx == 5:
                self.mesh[ix, iy, idx] = rb

    def gen_mesh(self):

        mesh = Mesh()

        for ix in range(self.ixmax-1):
            for iy in range(self.iymax-1):
                lb = self.gen_point(ix, iy)
                lt = self.gen_point(ix, iy+1)
                rt = self.gen_point(ix+1, iy+1)
                rb = self.gen_point(ix+1, iy)
                trlb = Triangle(lb, rb, lt)
                trrt = Triangle(rt, lt, rb)
                mesh.add_triangle(trlb)
                mesh.add_triangle(trrt)
 
        return mesh

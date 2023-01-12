import taichi.math as tm
import numpy as np
from ti_raytracer.world import Triangle, Mesh


class Terrain:
    def __init__(self, ixmax=4, iymax=4, usize=0.2):
        self.ixmax = ixmax
        self.iymax = iymax
        self.usize = usize
        self.height_map = np.ones(dtype=np.float32, shape=[ixmax, iymax])

    def sample(self, sample_func):
        for ix in range(self.ixmax):
            for iy in range(self.iymax):
                x = (ix - (self.ixmax/2))*self.usize
                y = (iy - (self.iymax/2))*self.usize
                self.height_map[ix][iy] = sample_func(x, y)

    def gen_point(self, ix, iy):
        x = (ix - (self.ixmax/2))*self.usize
        y = (iy - (self.iymax/2))*self.usize
        return tm.vec3(x, y, self.height_map[ix][iy])

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

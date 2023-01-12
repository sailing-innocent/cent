import taichi.math as tm
from ti_raytracer.world import Triangle, Mesh


def gen_rectangle_mesh(lb, rb, lt, rt, mesh):
    # print("is generating mesh for lb, lt, rt, rb: ", lb, lt, rt, rb)
    trlb = Triangle(lb, rb, lt)
    trrt = Triangle(rt, lt, rb)
    mesh.add_triangle(trlb)
    mesh.add_triangle(trrt)


class Block:
    def __init__(self, x=0.0, y=0.0, z=0.0, ux=2.0, uy=2.0, uz=2.0):
        self.x = x
        self.y = y
        self.z = z
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.indices = [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
        ]
        self.planes = [
            [0, 1, 2, 3],  # front
            [4, 5, 1, 0],  # left
            [7, 6, 5, 4],  # back
            [3, 2, 6, 7],  # right
            [4, 0, 3, 7],  # buttom
            [1, 5, 6, 2]  # top
        ]

    def gen_mesh_s(self, s):
        mesh = Mesh()
        idx = self.planes[s]
        lbidx = self.indices[idx[0]]
        ltidx = self.indices[idx[1]]
        rtidx = self.indices[idx[2]]
        rbidx = self.indices[idx[3]]
        lb = tm.vec3(
            self.x+self.ux*lbidx[0], self.y+self.uy*lbidx[1], self.z+self.uz*lbidx[2])
        lt = tm.vec3(
            self.x+self.ux*ltidx[0], self.y+self.uy*ltidx[1], self.z+self.uz*ltidx[2])
        rt = tm.vec3(
            self.x+self.ux*rtidx[0], self.y+self.uy*rtidx[1], self.z+self.uz*rtidx[2])
        rb = tm.vec3(
            self.x+self.ux*rbidx[0], self.y+self.uy*rbidx[1], self.z+self.uz*rbidx[2])
        gen_rectangle_mesh(lb, rb, lt, rt, mesh)
        return mesh

    def gen_mesh(self):
        mesh = Mesh()
        for i in range(6):
            mesh.append_mesh(self.gen_mesh_s(i))
        return mesh


class Blocks:
    def __init__(self, ix=1, iy=1, iz=1, ux=1, uy=1, uz=1):
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.ux = ux
        self.uy = uy
        self.uz = uz

    def gen_mesh(self):
        for x in range(self.ix):
            for y in range(self.iy):
                for z in range(self.iz):
                    block = Block()

import numpy as np
import open3d as o3d

print("Testing mesh in Open3D...")

armadillo_mesh = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
print("Vertices")
print(np.asarray(mesh.vertices))
print("Triangles")
print(np.asarray(mesh.triangles))
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

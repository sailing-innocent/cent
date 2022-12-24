import numpy as np 
import cv2 as cv
import os
import re
import scipy.io as sio

Train400IMG_PATH = "E:/data/datasets/Make3D/Train400Img"
Train400DEPTH_PATH = "E:/data/datasets/Make3D/Train400Depth"

img_names = os.listdir(Train400IMG_PATH)
test_img_name = img_names[1]
test_img_path = os.path.join(Train400IMG_PATH, test_img_name)
print(test_img_path)
img_code_res = re.findall(r"img-(.+).jpg", test_img_name)
print(img_code_res)
depth_names = os.listdir(Train400DEPTH_PATH)
test_depth_name = depth_names[1]
test_depth_path = os.path.join(Train400DEPTH_PATH, test_depth_name)
depth_code_res = re.findall(r"depth_sph_corr-(.+).mat", test_depth_name)
print(depth_code_res)
print(test_depth_path)

# debug the data

test_img = cv.imread(test_img_path)
cv.imshow("test img", test_img)

test_depth = sio.loadmat(test_depth_path)
key_names = list(test_depth.keys())
print(key_names)
key_name = key_names[-1]
test_depth_data = test_depth[key_name]
print(test_depth_data.shape)
test_data = np.zeros((55,305))

for row in range(len(test_depth_data)):
    for col in range(len(test_depth_data[row])):
        test_data[row][col] = test_depth_data[row][col][3]/80

test_data = cv.resize(test_data, (2272,1740))
print(test_data.shape)
cv.imshow("depth image", test_data)

"""
minx = 100000000000
maxx = -100000000000
miny = 100000000000
maxy = -100000000000
minz = 100000000000
maxz = -100000000000
mind = 100000000000
maxd = -100000000000
for row in range(len(test_depth_data)):
    for col in range(len(test_depth_data[row])):
        # print(test_depth_data[row][col])
        testdata = test_depth_data[row][col]
        minx = min(minx, testdata[0])
        maxx = max(maxx, testdata[0])
        miny = min(miny, testdata[1])
        maxy = max(maxy, testdata[1])
        minz = min(minz, testdata[2])
        maxz = max(maxz, testdata[2])
        mind = min(mind, testdata[3])
        maxd = max(maxd, testdata[3])

print("MIN X: ", minx)
print("MAX X: ", maxx)
print("MIN Y: ", miny)
print("MAX Y: ", maxy)
print("MIN Z: ", minz)
print("MAX Z: ", maxz)
print("MIN D: ", mind)
print("MAX D: ", maxd)

"""

cv.waitKey(0)
cv.destroyAllWindows()


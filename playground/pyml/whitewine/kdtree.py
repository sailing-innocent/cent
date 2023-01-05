# KD tree
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DBGMINX = 0.0
DBGMAXX = 10.0
DBGMINY = 0.0
DBGMAXY = 10.0

def pickSecond(elem):
    return elem[1]

def distance(x, y, p = 2):
    n = len(x)
    res = 0

    if (p==2):
        for i in range(n):
            res = res + (x[i] - y[i])**2
        return np.sqrt(res)

def Mid(xs, i):
    # return the mid item of i column
    temp = np.sort(np.array(xs)[:,i])
    N = len(xs)
    if (N % 2 == 1):
        return temp[N//2]
    else:
        return (temp[N//2-1]+temp[N//2])/2

class KDNode:
    def __init__(self, x_samples, depth = 0):
        self.parent = None
        self.leftChild = None
        self.rightChild = None
        self.nodes = []
        self.depth = depth
        self.axis = 0
        self.criteria = 0
        N = len(x_samples)
        # print("is creating KDNode for: \n", x_samples)
        self.leftarr = []
        self.rightarr = []
        if (N > 0):            
            # N > 0, we need partition
            dim = len(x_samples[0]) - 1
            # print("dim: ", dim)
            # print("node depth: ", depth)
            axis = (depth % dim)
            self.axis = axis
            # print("is splitting: ", axis)
            mid = Mid(x_samples, axis)
            self.criteria = mid

            for x in x_samples:
                self.nodes.append(x)
                # print("IS COMPARING {} and {}".format(x[axis], mid))
                if x[axis] < mid:
                    self.leftarr.append(x)
                elif x[axis] > mid:
                    self.rightarr.append(x)

            self.setLeft()
            self.setRight()

    def setLeft(self):
        # print("IS CREATING LEFT NODES")
        self.leftChild = KDNode(self.leftarr, self.depth+1)
        self.leftChild.parent = self

    def setRight(self):
        # print("IS CREATING RIGHT NODES")
        self.rightChild = KDNode(self.rightarr, self.depth+1)
        self.rightChild.parent = self

    def __str__(self):
        return self.nodes.__str__()

    def debug(self, ax, isLeft=False):            
        self.minx = DBGMINX
        self.maxx = DBGMAXX
        self.miny = DBGMINY
        self.maxy = DBGMAXY
        if (self.parent is not None):
            self.minx = self.parent.minx
            self.maxx = self.parent.maxx
            self.miny = self.parent.miny
            self.maxy = self.parent.maxy
            if (self.parent.axis == 0):
                if isLeft:
                    self.maxx = self.parent.criteria
                else:
                    self.minx = self.parent.criteria
            else:
                if isLeft:
                    self.maxy = self.parent.criteria
                else:
                    self.miny = self.parent.criteria

        rectangle = patches.Rectangle(
            (self.minx, self.miny),
            self.maxx - self.minx,
            self.maxy - self.miny,
            edgecolor = 'blue',
            fill = False
        )

        ax.add_patch(rectangle)
        if (self.leftChild is not None and self.rightChild is not None):
            self.leftChild.debug(ax, True)
            self.rightChild.debug(ax)

class KDTree(KDNode):
    def __init__(self, x_samples):
        # print("IS CONSTRUCTING KDTREE FOR \n {}".format(x_samples))
        super(KDTree, self).__init__(x_samples)

    def debugTree(self, ax):
        # print("Debugging KDTree")
        dbg = np.array(self.nodes)
        tags = dbg[:,0]
        labels = []
        for tag in tags:
            labels.append([0.2, 0.8, 0.3])
        ax.scatter(dbg[:,0], dbg[:,1], c = labels)
        super(KDTree, self).debug(ax)

def find_k_nearest_kdt(kdtree, x_test, k = 3):
    visited = []
    k_nearest = []
    # k_nearest = [[2,3,0],[8,1,1],[7,2,1],]
    # construct kdtree

    # find the right kd node where the data belongs to
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("is finding k-nearest for ", x_test)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # forward find the nearest node
    node = kdtree 

    while (node.leftChild is not None and node.rightChild is not None):
        # print("is searching {} \n among \n {}".format(x_test, node))
        if (x_test[node.axis] < node.criteria):
            node = node.leftChild
        else:
            node = node.rightChild
    # print(node.axis, node.criteria)
    curr = node.parent
    while (curr is not None):
        # print("current nodes: ", curr.nodes)
        for item in curr.nodes:
            isDup = False
            for v in visited:
                # if already in, continue
                if v == item[-1]:
                    isDup = True
            if (isDup):
                continue
            else:
                visited.append(item[-1])
            
            d = distance(item[0:-1], x_test)
            # print("COMPARING {} and {} got {}".format(item, x_test, d))
            # print("==========")
            
            if (len(k_nearest) < k):
                k_nearest.append([item[-1], d])
                k_nearest.sort(key=pickSecond)
            else:
                if k_nearest[k-1][1] > d:
                    k_nearest[k-1] = [item[-1],d]
                    k_nearest.sort(key=pickSecond)
        curr = curr.parent
    return k_nearest

def kdknn(kdtree, y_samples, x_test, k = 3):
    k_nearest = find_k_nearest_kdt(kdtree, x_test, k)
    # print("GET K-Nearest: ", k_nearest)
    catedict = dict()
    for k_item in k_nearest:
        # print(y_samples[k_item[0]])
        cate = str(int(y_samples[k_item[0]]))
        if cate in catedict.keys():
            catedict[cate] = catedict[cate] + 1
        else:
            catedict[cate] = 1
        
    maxval = 0
    res = 0
    for cate in catedict.keys():
        if catedict[cate] > maxval:
            maxval = catedict[cate]
            res = cate
    if (res == 0):
        print("SOMETHING WRONG: ", catedict)
    return res

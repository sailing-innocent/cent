# KD tree
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DBGMINX = 0.0
DBGMAXX = 10.0
DBGMINY = 0.0
DBGMAXY = 10.0

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
        N = len(x_samples)
        print("is creating KDNode for: \n", x_samples)
        if (N > 0):            
            # N > 0, we need partition
            dim = len(x_samples[0]) - 1
            print("dim: ", dim)
            print("node depth: ", depth)
            axis = depth % dim
            self.axis = axis
            print("is splitting: ", axis)
            mid = Mid(x_samples, axis)
            self.criteria = mid
            leftarr = []
            rightarr = []
            for x in x_samples:
                self.nodes.append(x)
                print("IS COMPARING {} and {}".format(x[axis], mid))
                if x[axis] < mid:
                    leftarr.append(x)
                elif x[axis] > mid:
                    rightarr.append(x)

            self.setLeft(leftarr)
            self.setRight(rightarr)

    def setLeft(self, x_samples):
        print("IS CREATING LEFT NODES")
        self.leftChild = KDNode(x_samples, self.depth+1)
        self.leftChild.parent = self

    def setRight(self, x_samples):
        print("IS CREATING RIGHT NODES")
        self.rightChild = KDNode(x_samples, self.depth+1)
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
        print("IS CONSTRUCTING KDTREE FOR \n {}".format(x_samples))
        super(KDTree, self).__init__(x_samples)

    def debugTree(self, ax):
        print("Debugging KDTree")
        dbg = np.array(self.nodes)
        tags = dbg[:,2]
        labels = []
        for tag in tags:
            labels.append([0.2, 0.8 * tag, 0.3])
        ax.scatter(dbg[:,0], dbg[:,1], c = labels)
        super(KDTree, self).debug(ax)
    



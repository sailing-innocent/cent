from knn import *
from kdtree import *


def gen_samples():
    return [[2,3],[5,4],[6,6],[4,7],[8,1],[7,2],[8,5],[8,3]],[0,0,0,0,1, 1,1,1]

def test_distance():
    x = np.array([1,2])
    y = np.array([2,1])
    print(distance(x,y))

def debug_kdtree(kdt, ax):
    ax.plot([1,4],[1,4])
    ax.add_patch(
        patches.Rectangle(
            (1,1),
            0.5,
            0.5,
            edgecolor = 'blue',
            facecolor = 'red',
            fill = True
        )
    )

def debug_samples(ax, x, y):
    N = len(x)
    dbgx = []
    dbgy = []
    dbgc = []
    for i in range(N):
        dbgx.append(x[i][0])
        dbgy.append(x[i][1])
        dbgc.append([0.0, y[i] * 0.4 + 0.4, 0.5])
    ax.scatter(dbgx, dbgy, c = dbgc)

def debug_tests(ax, x, y):
    
    dbgx = [x[0]]
    dbgy = [x[1]]
    dbgc = [[0.0, float(y)*1.0, 0.5]]
    ax.scatter(dbgx, dbgy, c = dbgc)

def debug_knn(x_samples, y_samples, x, y):
    fix, ax = plt.subplots()
    debug_samples(ax, x_samples, y_samples)
    debug_tests(ax, x, y)
    plt.show()

def debug_kdknn(kdtree):
    print("++++++++++++++++++++ CHECKING KDTREE ++++++++++++++++ ")
    # print(kdtree)
    # print(kdtree.leftChild)
    # print(kdtree.rightChild)
    # print(kdtree.rightChild.parent)
    fix, ax = plt.subplots()
    kdtree.debugTree(ax)
    plt.show()

def utest():
    """KDKNN
    print("unit testing")
    x_samples, y_samples = gen_samples()
    x_test = np.array([5, 7])
    # x_test = np.array([6, 3])
    label = knn(x_samples, y_samples, x_test)
    debug_knn(x_samples, y_samples, x_test, label)
    """

    x_samples, y_samples = gen_samples()
    x_test = [5,5]
    label = kdknn(x_samples, y_samples, x_test)
    debug_knn(x_samples, y_samples, x_test, label)
    # x_samples = gen_kdtree_samples()
    # kdtree = KDTree(x_samples)
    # debug_kdknn(kdtree)


if __name__ == "__main__":
    utest()
    plt.show()


import numpy as np
import os
os.chdir("C:\\Studying\\GrowthHackers\\Cheatsheet")

# for help, np.info(np.ndarray)
# array is a functino to create an ndarray, and ndarray itself is a class

matx= np.array([[(1.5, 2, 3), (0, 1, 2)], [(3, 2, 1), (4, 5, 6)]])
#3d matrix
# print(matx)
# print(matx.shape, len(matx), matx.ndim, matx.size)
# #len() gets the first dimention's number, size gets the multiplication of array
# matx= matx.astype(bool)
# print(matx, matx.dtype)
#
# #placeholding by zero matrix, one matrix, arange, linspace and else
# print(np.zeros((2, 5)))
# print(np.ones((4, 3)))
# print(np.arange(10, 23, 3))
# print(np.linspace(0, 2, 21))
# # difference between arange: 3rd argument- arange means counting value,
# # and linspace means counting number
#
# print(np.full((2,2), 7))
# print(np.eye(3))
# print(np.random.random((2,2)))
# print(np.empty((3,2)))

matx1= matx[0,:,:]
matx2= matx[1,:,:]
matx3= matx[0, :, 1]

# print(matx1+matx2)
# print(matx1*matx2, matx1/ matx2)        #not a matrix operation
# print(np.exp(matx1))
# print(np.dot(matx1[0,:], matx2[0,:]))
# print(np.dot(matx1, matx2.T))
#
# print(matx.sum(), matx.min(), matx.mean(), np.std(matx))
#
# matx_copy= matx.copy()       #deep copy
# matx_empty= np.empty_like(matx)
# matx_broadcast= np.tile(matx3, (3, 2))
# print(matx_empty)
# print(matx_broadcast)
# print(matx[[1,0,1,0], [0,1,0,0], [1,2,0,1]])
# print(matx>=2)
#
# print(matx.ravel())     # or, print(matx.flatten())
# print(matx.reshape(4, -3))

# print(matx1)
# print(np.append(matx1, matx2))
# print(np.insert(matx1, 1, 5))
# print(np.delete(matx1, 2))
# print(matx1)
# # append, insert automaticly make matrix a 1-D vector(np.ravel())
# # method do not change original value
#
# print(np.concatenate((matx1, matx2), axis=1))
# print(np.vstack((matx1[0, :], matx[1, :], matx2[0, :])))
# print(np.hstack((matx1[0, :], matx2[0, :], matx2[1, :])))       # or, print(np.r_[3, [0]*5, -1:1:11j])
# print(np.column_stack((matx1[0, :], matx2[0, :])))        # or, print(np.c_[matx1[0, :], matx2[0, :], matx2[1, :]])
# print(np.column_stack((matx1, matx2[:, 0])))        # if vector then transpose, else not.
#
# print(matx)
# print(np.r_[3, [0]*5, -1:1:11j])
# print(np.c_[matx1[0, :], matx2[0, :], matx2[1, :]])
# print(np.hsplit(matx1, 3))
# print(np.vsplit(matx1, 2))
#
# print(np.mgrid[0:5, 0:5])       #Dense meshgrid
# print(np.ogrid[0:5, 0:5])       #Open meshgrid

# def myfunc(a):
#     if a<0:
#         return a*2
#     else:
#         return a/2
#
# vfunc= np.vectorize(myfunc, otypes= [np.float])
# print(vfunc([-3, -1, 1, 3]))

g= np.linspace(0, np.pi, num=5)
g[3:] += np.pi
print(g)
print(np.unwrap(g))

c = np.array([[(1.5,2,3), (2,3,4)], [(3,2,1), (3,4,5)]])
print(np.select([c<4],[c*2]))

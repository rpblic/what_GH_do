import numpy as np
from scipy import linalg, sparse
# scipy.linalg contains and expands on numpy.linalg

matxA= np.matrix(np.random.random((3,3)))
matxB= np.asmatrix([(1+5j,2j,3j), (4j,5j,6j)])      # same as copy= False. reference or make.
matxC= np.mat(np.random.random((10, 5)))
matxD= np.mat([[3, 4], [5, 6]])
print(matxA)
print(matxB)
print(matxC)
print(matxD)
#
# print(matxA.T)        # Trace: 대각합
# print(matxA.H)      # Conjugate Transposition. About imaginary matrix
# print(matxA.I)      # or, linalg.inv(matxA)
# print(np.trace(matxA))
# print(linalg.norm(matxA))       # Frobenius norm, https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
# print(linalg.norm(matxA, l))        #L1 norm
print(np.linalg.matrix_rank(matxC))

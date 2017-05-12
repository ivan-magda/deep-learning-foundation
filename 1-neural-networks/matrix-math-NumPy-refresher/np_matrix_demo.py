
import numpy as np


######################################################################################################
# NumPy Matrix multiplication 																		 #
######################################################################################################
#
# You've heard a lot about matrix multiplication in the last few videos – now you'll get to see how to
# do it with NumPy. However, it's important to know that NumPy supports several types of matrix multiplication.
#
# Element-wise Multiplication
# You saw some element-wise multiplication already. You accomplish that with the multiply function or the * operator.
#  Just to revisit, it would look like this:

m = np.array([[1,2,3],[4,5,6]])
print(m)
# displays the following result:
# array([[1, 2, 3],
#        [4, 5, 6]])

n = m * 0.25
print(n)
# displays the following result:
# array([[ 0.25,  0.5 ,  0.75],
#        [ 1.  ,  1.25,  1.5 ]])

print(m * n)
# displays the following result:
# array([[ 0.25,  1.  ,  2.25],
#        [ 4.  ,  6.25,  9.  ]])

print(np.multiply(m, n))   # equivalent to m * n
# displays the following result:
# array([[ 0.25,  1.  ,  2.25],
#        [ 4.  ,  6.25,  9.  ]])

# Matrix Product
# To find the matrix product, you use NumPy's matmul function.
#
# If your have compatible shapes, the it's as simple as this:

a = np.array([[1,2,3,4],[5,6,7,8]])
print(a)
# displays the following result:
# array([[1, 2, 3, 4],
#        [5, 6, 7, 8]])

print(a.shape)
# displays the following result:
# (2, 4)

b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(b)
# displays the following result:
# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])

print(b.shape)
# displays the following result:
# (4, 3)

c = np.matmul(a, b)
print('A matmul B:\n', c)
# displays the following result:
# array([[ 70,  80,  90],
#        [158, 184, 210]])

print(c.shape)
# displays the following result:
# (2, 3)


# NumPy's dot function
# You may sometimes see NumPy's dot function in places where you would expect a matmul. 
# It turns out that the results of dot and matmul are the same if the matrices are two dimensional.
#
# So these two results are equivalent:

a = np.array([[1,2],[3,4]])
print(a)
# displays the following result:
# array([[1, 2],
#        [3, 4]])

np.dot(a, a)
# displays the following result:
# array([[ 7, 10],
#        [15, 22]])

print(a.dot(a))  # you can call `dot` directly on the `ndarray`
# displays the following result:
# array([[ 7, 10],
#        [15, 22]])

print(np.matmul(a,a))
# array([[ 7, 10],
#        [15, 22]])

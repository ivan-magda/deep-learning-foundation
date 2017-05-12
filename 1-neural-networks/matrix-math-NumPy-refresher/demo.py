
import numpy as np

######################################################################################################
# Scalars 																							 #
######################################################################################################
#
# Scalars in NumPy are a bit more involved than in Python. 
# Instead of Python’s basic types like int, float, etc., NumPy lets you specify signed and unsigned types, as well as
# different sizes. So instead of Python’s int, you have access to types like uint8, int8, uint16, int16, and so on.

scalar = np.array(5)
print('Scalar = ', scalar, 'shape: ', scalar.shape)

scalarSum = scalar + 11
print('Scalar sum: ', scalar, ' + 11 = ', scalarSum)


######################################################################################################
# Vectors 																							 #
######################################################################################################
#
# To create a vector, you'd pass a Python list to the array function, like this:

vector = np.array([1, 2, 3]) 
print('Vector: ', vector, 'shape: ', vector.shape)

# Access an element within the vector using indices, like this:
print('vector[1] = ', vector[1])

# NumPy also supports advanced indexing techniques.
# For example, to access the items from the second element onward, you would say:
print(vector[1:])


######################################################################################################
# Matrices 																							 #
######################################################################################################
#
# You create matrices using NumPy's array function, just you did for vectors.
# However, instead of just passing in a list, you need to supply a list of lists, where each list represents a row.
# So to create a 3x3 matrix containing the numbers one through nine, you could do this:

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('Matrix: \n', matrix)
print('Matrix shape: ', matrix.shape)
print('matrix[1][2] = ', matrix[1][2])


######################################################################################################
# Tensors 																							 #
######################################################################################################
#
# Tensors are just like vectors and matrices, but they can have more dimensions. 
# For example, to create a 3x3x2x1 tensor, you could do the following:

tensor = np.array([[[[1],[2]],[[3],[4]],[[5],[6]]],[[[7],[8]],\
    [[9],[10]],[[11],[12]]],[[[13],[14]],[[15],[16]],[[17],[18]]]])

print('Tensor: \n', tensor)
print('Tensor shape: ', tensor.shape)

# Access items just like with matrices, but with more indices. So t[2][1][1][0] will return 16.
print('tensor[2][1][1][0]', tensor[2][1][1][0])

######################################################################################################
# Element-wise operations																			 #
######################################################################################################
# The Python way
# Suppose you had a list of numbers, and you wanted to add 5 to every item in the list. 
# Without NumPy, you might do something like this:

p_values = [1, 2, 3, 4, 5]
for i in range(len(p_values)):
    p_values[i] += 5

# now values holds [6, 7, 8, 9, 10]

# The NumPy way
# In NumPy, we could do the following:

np_values = [1, 2, 3, 4, 5]
np_values = np.array(np_values) + 5

print('[1, 2, 3, 4, 5] + 5 == ', np_values)

# now values is an ndarray that holds [6, 7, 8, 9, 10]

# We should point out, NumPy actually has functions for things like adding, multiplying, etc.
# But it also supports using the standard math operators. So the following two lines are equivalent:

some_array = np.array([1, 2, 3, 4, 5])
np_mult_1 = np.multiply(some_array, 5)
np_mult_2 = some_array * 5

print('[1, 2, 3, 4, 5] * 5 == ', np_mult_2)

# Init with zeros:
zero_matrix = np_mult_2 * 0

# now every element in m is zero, no matter how many dimensions it has
print(np_mult_2, '* 0 == ', zero_matrix)


######################################################################################################
# Element-wise Matrix Operations 																	 #
######################################################################################################
#
#The same functions and operators that work with scalars and matrices also work with other dimensions.
# You just need to make sure that the items you perform the operation on have compatible shapes.
#
# Let's say you want to get the squared values of a matrix. 
# That's simply x = m * m (or if you want to assign the value back to m, it's just m *= m
#
# This works because it's an element-wise multiplication between two identically-shaped matrices.
# (In this case, they are shaped the same because they are actually the same object.)
#
# Here's the example:

a = np.array([[1, 3], [5, 7]])
print('A:\n', a)
# displays the following result:
# array([[1, 3],
#        [5, 7]])

b = np.array([[2, 4], [6, 8]])
print('B:\n', b)
# displays the following result:
# array([[2, 4],
#        [6, 8]])

print('A + B:\n', a + b)
# displays the following result
#      array([[ 3,  7],
#             [11, 15]])

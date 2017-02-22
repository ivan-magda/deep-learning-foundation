# Quiz Solution
# Note: You can't run code in this tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.cast(tf.divide(x,y), tf.int32),tf.constant(1))

# Note:TensorFlow has multiple ways to divide.
#   tf.divide(x,y) uses Python 3 division semantics and will return a float here
#          It would be the best choice if all the other values had been floats
#   tf.div(x,y) uses Python 2 division semantics and will return an integer here
#          TensorFlow documentation suggests we should prefer tf.divide
#   tf.floordiv(x,y) will do floating point division and then round down to the nearest
#          integer (but the documentation says it may still represent
#          its result as a floating point value)
#   tf.cast(tf.divide(x,y), tf.int32)
#          This lets us do floating point division and then cast it to an integer
#          to match the 1 passed to subtract


# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)

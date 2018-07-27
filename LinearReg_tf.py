import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
rng = numpy.random

# Parameters
learning_rate = 1.0/10**8
training_epochs = 1000
display_step = 10
beta = 100
# Training Data
raw_X = read_csv("/home/vincent/machine_learning/LinearReg/data/X.csv", header=0)
train_X = numpy.asarray(raw_X.values)
raw_Y = read_csv("/home/vincent/machine_learning/LinearReg/data/Y.csv", header=0)
train_Y = numpy.asarray(raw_Y.values)
n_samples = train_X.shape[0]


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(tf.zeros([9]), name="weights")
b = tf.Variable(tf.zeros([1]), name="bias")

# Construct a linear model
pred = tf.add(tf.reduce_sum(tf.multiply(X, W)), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Loss function using L2 Regularization
regularizer = tf.nn.l2_loss(W)
cost = tf.reduce_mean(cost + beta * regularizer)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.AdagradOptimizer(learning_rate,initial_accumulator_value=0.1).minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    '''
    raw_X = read_csv("/home/vincent/machine_learning/LinearReg/data/test_x.csv", header=0)
    test_X = numpy.asarray(raw_X.values)
    for x in test_X:
        predicted_result = sess.run(tf.reduce_sum(pred), feed_dict={X: x})
        print(predicted_result)
    '''
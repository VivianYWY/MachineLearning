
# coding: utf-8

# In[6]:


import tensorflow as tf


# In[3]:


x = tf.Variable(3, name="x") 
y = tf.Variable(4, name="y")
f = x*x*y + y + 2


# In[4]:


init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run() # actually initialize all variables
    print(f.eval())


# In[5]:


w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    print(y.eval()) # evaluates w and x twice because node values are dropped between graph runs
    print(z.eval())


# In[7]:


with tf.Session() as sess:
    y_val, z_val = sess.run([y,z]) # evaluate both in just one graph run 
    print(y_val)
    print(z_val)


# # linear regression with tensorflow using Normal Equation

# In[10]:


import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)  # benefit is tensorflow will automatically run this on GPU if have one

with tf.Session() as sess:
    theta_value = theta.eval()


# In[11]:


theta_value


# compare with Numpy

# In[12]:


X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta_numpy)


# compare with sklearn

# In[13]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])


# # using Batch Gradient Descent

# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)   # Gradient Descent requires scaling the feature vectors first, tf also has scaling module
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]


# Manually computing the gradients

# In[20]:


n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()


# # Using autodiff

# Same as above except for the gradients = ... line:

# In[ ]:


gradients = tf.gradients(mse, [theta])[0]   # automatically compute the partial derivative for all variables


# # Using optimizer 

# Same as above except for the gradients = ... and training_op = ...line:

# In[21]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


# In[23]:


optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)


# # Feeding data to the training algorithm

# Placeholder nodes

# In[24]:


A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)


# Mini-batch Gradient Descent

# In[25]:


X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")


# In[26]:


theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()


# In[27]:


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


# In[28]:


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()


# In[29]:


best_theta


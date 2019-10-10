from data_prep import *
import numpy as np
import time

data = np.asarray(data)

def relu(value):
    return np.maximum(0,value)

def relue_prime(value):
    return np.where(value <= 0, 0, 1)

def tanh(value):
    return np.tanh(value)

def tanh_prime(value):
    return 1-np.tanh(value)**2


input_size = 8
n_h1 = 10
n_h2 = 10
n_o = 2
epoch = 10000
learning_rate = 0.1

# shape = 10 x 8
w_i_h1 = np.random.normal(0,1,(input_size,n_h1))
b_i_h1 = np.random.normal(0,1,n_h1)
#  shape = 10 x 10
w_h1_h2 = np.random.normal(0,1,(n_h1,n_h2))
b_h1_h2 = np.random.normal(0,1,n_h2)
#  shape = 2 x 10
w_h2_out = np.random.normal(0,1,(n_h2,n_o))
b_h2_out = np.random.normal(0,1,n_o)


#       w_i_h1          relu          w_h1_h2         relu         w_h2_out      tanh           MSE
#   i--------->h1_in--------->h1_out---------->h2_in------->h2_out--------->o_in--------->out--------->err
for j in range(epoch):
    error = 0
    start = time.time()
    for i,y1,y2 in zip(data,c_users,r_users):

        y = [y1,y2]
        h1_in = np.dot(i,w_i_h1)+b_i_h1
        h1_out = relu(h1_in)
        h2_in = np.dot(h1_out,w_h1_h2)+b_h1_h2
        h2_out = relu(h2_in)
        o_in = np.dot(h2_out,w_h2_out)+b_h2_out
        out = tanh(o_in)

        d_err_out = y - out
        error = d_err_out
        d_out_o_in = tanh_prime(o_in)
        d_err_o_in = d_err_out * d_out_o_in
        d_o_in_w_h2_out = h2_out
        d_o_in_h2_out = w_h2_out

        d_err_w_h2_out = np.dot(d_o_in_w_h2_out[:,None],d_err_o_in[:,None].T)
        d_err_b_h2_out = d_err_o_in
        
        d_err_h2_out = np.dot(d_o_in_h2_out,d_err_o_in)
        d_h_out_h2_in = relue_prime(h2_in)
        d_err_h2_in = d_err_h2_out * d_h_out_h2_in
        d_h2_in_w_h1_h2 = h1_out
        d_h2_in_h1_out = w_h1_h2

        d_err_w_h1_h2 = np.dot(d_h2_in_w_h1_h2[:,None],d_err_h2_in[:,None].T)
        d_err_b_h1_h2 = d_err_h2_in
        
        d_err_h1_out = np.dot(d_h2_in_w_h1_h2,d_err_h2_in)
        d_h1_out_h1_in = relue_prime(h1_in)
        d_err_h1_in = d_err_h1_out * d_h1_out_h1_in
        d_h1_in_w_i_h1 = i
        d_h1_in_i = w_i_h1

        d_err_w_i_h1 = np.dot(d_h1_in_w_i_h1[:,None],d_err_h1_in[:,None].T)
        d_err_b_i_h1 = d_err_h1_in

        w_i_h1 -= learning_rate * d_err_w_i_h1
        b_i_h1 -= learning_rate * d_err_b_i_h1

        w_h1_h2 -= learning_rate * d_err_w_h1_h2
        b_h1_h2 -= learning_rate * d_err_b_h1_h2

        w_h2_out -= learning_rate * d_err_w_h2_out
        b_h2_out -= learning_rate * d_err_b_h2_out

    print(time.time() - start,end='\r')

for _ in range(10):

        n = np.random.randint(100,200)
        i = data[n]
        y = [c_users[n],r_users[n]]
        h1_in = np.dot(i,w_i_h1)+b_i_h1
        h1_out = relu(h1_in)
        h2_in = np.dot(h1_out,w_h1_h2)+b_h1_h2
        h2_out = relu(h2_in)
        o_in = np.dot(h2_out,w_h2_out)+b_h2_out
        out = tanh(o_in)
        print(y)
        print(out)

        



import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow as tf


path_train="C:/Anaconda codes/stock market prediction/csv/Gold Data Last year.csv"
path_test="C:/Anaconda codes/stock market prediction/csv/Gold Data Last Month.csv"


current_train=path_train
current_test=path_test

num_train=266
num_test=22
lr=0.1
epochs=100



def load_data(stock_name,num_data_points):
    data=pd.read_csv(stock_name,
                    skiprows=0,
                    nrows=num_data_points,
                    usecols=['Price','Open','Vol.'])
    final_price=data['Price'].astype(str).str.replace(',','').astype(np.float)
    open_price=data['Open'].astype(str).str.replace(',','').astype(np.float)
    vol=data['Vol.'].str.strip('MK').astype(np.float)
    
    return final_price,open_price,vol
#--------------------------------------------------------------------

def cal_diff(final,opens):
    diff=[]
    
    for i in range(len(final)-1):
        diff.append(opens[i+1]-final[i])
    return(diff)

#--------------------------------------------------------------------
def calculate_accuracy(expected_values, actual_values):
    num_correct = 0
    for a_i in range(len(actual_values)):
        if actual_values[a_i] < 0 < expected_values[a_i]:
            num_correct += 1
        elif actual_values[a_i] > 0 > expected_values[a_i]:
            num_correct += 1
    return (num_correct / len(actual_values)) * 100
#-------------------------------------------------------------------

#train data
train_final,train_openings,train_vol=(load_data(current_train,num_train))
train_difference = cal_diff(train_final,train_openings)
train_vol=train_vol[:-1]

#test data
test_final,test_openings,test_vol=(load_data(current_test,num_test))
test_difference = cal_diff(test_final,test_openings)
test_vol=test_vol[:-1]



#y = Wx + b

x=tf.placeholder(tf.float32, name='x')
W=tf.Variable([0.1],name='W')
b=tf.Variable([0.1],name='b')
y=W*x+b

y_predicted=tf.placeholder(tf.float32, name='x')\

loss=tf.reduce_sum(tf.square(y-y_predicted))
optimizer=tf.train.AdamOptimizer(lr).minimize(loss)

session=tf.Session()
session.run(tf.global_variables_initializer())

for _ in range(epochs):
    session.run(optimizer, feed_dict={x:train_vol, y_predicted:train_difference})
    
    
results = session.run(y, feed_dict={x: test_vol})
accuracy = calculate_accuracy(test_difference, results)
print("Accuracy of model: {0:.2f}%".format(accuracy))


print(session.run(y,{x:[200,500,340]}))


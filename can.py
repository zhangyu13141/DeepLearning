import numpy as np
np.random.seed(1)
#nonlinear function
def sigmoid(X,derive=False):
    if not derive:
        return 1/(1+np.exp(-X))
    return X*(1-X)
def relu(X,derive=False):
    if not derive:
        return np.maximum(0,X)
    return(X>0).astype(float)
nonline=relu#nonline=sigmoid

#input
X=np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
#output
y=np.array([[0],
            [1],
            [1],
            [0]])
#weight,bais
W1=2*np.random.random((3,4))-1
b1=0.1*np.ones((4,))
W2=2*np.random.random((4,1))-1
b2=0.1*np.ones((1,))
#training
train_times=60000
for time in range(train_times):
    #layer1
    A1=np.dot(X,W1)+b1
    Z1=nonline(A1)
    #layer2
    A2=np.dot(Z1,W2)+b2
    _y=Z2=nonline(A2)
    cost=_y-y
    print('Error{}'.format(np.mean(np.abs(cost))))
    #Calc deltas
    delta_A2=cost*nonline(Z2,derive=True)#noline导
    delta_b2=delta_A2.sum(axis=0)
    delta_W2=np.dot(Z1.T,delta_A2)
    delta_A1=np.dot(delta_A2,W2.T)*nonline(Z1,derive=True)
    delta_b1=delta_A1.sum(axis=0)
    delta_W1=np.dot(X.T,delta_A1)
    #apply deltas
    rate=0.11
    W1-=rate*delta_W1
    b1-=rate*delta_b1
    W2-=rate*delta_W2
    b2-=rate*delta_b2
else:
    print('Predict:')
    print(_y)





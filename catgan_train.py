#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import numpy as np
import six
import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

import data

parser = argparse.ArgumentParser(description='CVAE')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=200, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')

args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255


N = 60000
nz=74
#N_test = y_test.size
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
#x_train=x_train[0:5000,:]


#N=5000

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy




class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0=L.Linear(128,500),
            l1=L.Linear(500,500),
            l2=L.Linear(500,1000),
            l3=L.Linear(1000,784),
            bn0 = L.BatchNormalization(500),
            bn1 = L.BatchNormalization(500),
            bn2 = L.BatchNormalization(1000)
        )
        
    def __call__(self, z, test=False):
        h1=F.leaky_relu(self.bn0(self.l0(z),test),slope=0.1)
        h2=F.leaky_relu(self.bn1(self.l1(h1),test),slope=0.1)
        h3=F.leaky_relu(self.bn2(self.l2(h2),test))
        h4=F.sigmoid(self.l3(h3))

        return h4

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            l0=L.Linear(784,1000),
            l1=L.Linear(1000,500),
            l2=L.Linear(500,250),
            l3=L.Linear(250,250),
            l4=L.Linear(250,250),
            l5=L.Linear(250,10),
            bn0 = L.BatchNormalization(1000),
            bn1 = L.BatchNormalization(500),
            bn2 = L.BatchNormalization(250),
            bn3 = L.BatchNormalization(250),
            bn4 = L.BatchNormalization(250),
        )
        
    def __call__(self, x, test=False):

        mu_array1=chainer.Variable(xp.array(xp.zeros([batchsize,784]),dtype=np.float32))
        log_std_array1=chainer.Variable(xp.log(0.09)*xp.array(xp.ones([batchsize,784]),dtype=np.float32))

        mu_array2=chainer.Variable(xp.array(xp.zeros([batchsize,1000]),dtype=np.float32))
        log_std_array2=chainer.Variable(xp.log(0.09)*xp.array(xp.ones([batchsize,1000]),dtype=np.float32))

        mu_array3=chainer.Variable(xp.array(xp.zeros([batchsize,500]),dtype=np.float32))
        log_std_array3=chainer.Variable(xp.log(0.09)*xp.array(xp.ones([batchsize,500]),dtype=np.float32))

        mu_array4=chainer.Variable(xp.array(xp.zeros([batchsize,250]),dtype=np.float32))
        log_std_array4=chainer.Variable(xp.log(0.09)*xp.array(xp.ones([batchsize,250]),dtype=np.float32))

        mu_array5=chainer.Variable(xp.array(xp.zeros([batchsize,250]),dtype=np.float32))
        log_std_array5=chainer.Variable(xp.log(0.09)*xp.array(xp.ones([batchsize,250]),dtype=np.float32))

        mu_array6=chainer.Variable(xp.array(xp.zeros([batchsize,250]),dtype=np.float32))
        log_std_array6=chainer.Variable(xp.log(0.09)*xp.array(xp.ones([batchsize,250]),dtype=np.float32))

        x=x+F.gaussian(mu_array1,log_std_array1)
        h1=F.leaky_relu(self.bn0(self.l0(x)+F.gaussian(mu_array2,log_std_array2),test),slope=0.1)
        h2=F.leaky_relu(self.bn1(self.l1(h1)+F.gaussian(mu_array3,log_std_array3),test),slope=0.1)
        h3=F.leaky_relu(self.bn2(self.l2(h2)+F.gaussian(mu_array4,log_std_array4),test),slope=0.1)
        h4=F.leaky_relu(self.bn3(self.l3(h3)+F.gaussian(mu_array5,log_std_array5),test),slope=0.1)
        h5=F.leaky_relu(self.bn4(self.l4(h4)+F.gaussian(mu_array6,log_std_array6),test),slope=0.1)
        h6=F.softmax(self.l5(h5))


        return h6

def d_entropy1(y):
    y1=F.sum(y,axis=0)/batchsize
    y2=F.sum(-y1*F.log(y1))
    return y2
def d_entropy2(y):
    y1=-y*F.log(y)
    y2=F.sum(y1)/batchsize
    return y2

# Setup optimizer
gen = Generator()
dis = Discriminator()
o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
o_gen.setup(gen)
o_dis.setup(dis)
o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))




# Learning loop
error=np.zeros([n_epoch,3])
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_l_dis = np.float32(0)
    sum_l_gen = np.float32(0)
        
    for i in six.moves.range(0, N, batchsize):
        z = xp.random.uniform(0, 1,size=(batchsize,128))
        z=chainer.Variable(z.astype(xp.float32))
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))

        fake_x=gen(z)
        fake_y=dis(fake_x)
        real_y=dis(x)

        #train discriminator
        L_dis=-1*(d_entropy1(real_y)-d_entropy2(real_y)+d_entropy2(fake_y))#Equation (7) upper
        o_dis.zero_grads()
        L_dis.backward()    
        o_dis.update()


        #train generator
        L_gen=d_entropy1(fake_y)+d_entropy2(fake_y)#Equation (7) lower
       
        o_gen.zero_grads()  
        L_gen.backward()
        o_gen.update()


        sum_l_dis+=L_dis.data
        sum_l_gen+=L_gen.data



    error[epoch-1,:]=[epoch,sum_l_dis,sum_l_gen]
    
    print('dis_loss',sum_l_dis,sum_l_gen,sum_l_dis+sum_l_gen)
#   print('loss',sum_l_gen)

np.savetxt('train_error.csv',error,delimiter=',',header='epoch,dis_loss,gen_loss')
# Save the model and the optimizer
print('save the model')
serializers.save_npz('catgan_gen.model', gen)
serializers.save_npz('catgan_dis.model', dis)



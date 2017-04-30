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
import matplotlib.pyplot as plt
import data

parser = argparse.ArgumentParser(description='CVAE')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=200, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--initmodel', '-m', default='catgan_gen.model',
                    help='Initialize the model from given file')
parser.add_argument('--fig_name', '-fig_name', default='tmp',
                    help='Output file name')

args = parser.parse_args()




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

gen = Generator()
serializers.load_npz(args.initmodel, gen)


name=args.fig_name
z = xp.random.uniform(0, 1,size=(1,128))
z=np.float32(z)
z[np.where(z>1)]=1
z[np.where(z<0)]=0

fake_x=gen(z,test=True).data
fake_x=fake_x.reshape(28,28)
plt.clf()             # flip vertical
plt.xlim(0,27)
plt.ylim(0,27)
plt.pcolor(fake_x)
plt.tick_params(labelbottom="off")
plt.tick_params(labelleft="off")
plt.gray()
fig_name='./fig/'+name+'.png'
plt.savefig(fig_name)


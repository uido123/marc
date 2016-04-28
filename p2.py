
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def idm(dim,width=0):
    M1 = np.tri(dim,dim,width)
    M2 = np.tri(dim,dim,-width-1)
    M3 = np.ones([dim,dim])
    M4 = M3 - M2
    M5 = np.multiply(M1,M4)
    return M5


# In[10]:

# total variation matrix
def TV(dim):
    M1 = np.eye(dim-1,dim,0)
    M2 = np.eye(dim-1,dim,1)
    return M1 - M2
# objective
def H(A,b,M):
    def h(u):
        return np.linalg.norm(np.dot(A,u)-b)**2 + np.sum(np.fabs(np.dot(M,u)))
    return h


# In[1]:

def load_lenna(a):
    from scipy import misc
    lenna = sp.misc.imread('lenna_bw.jpg')
    lenna_bw = lenna[a:-a,a:-a,0]
    return 0.1*lenna_bw/np.max(lenna_bw)


# In[65]:

def downscale(inp,r):
    d = int(inp.shape[0]/r)
    outp = np.zeros([d,d])
    for i in range(d):
        for j in range(d):
            outp[i,j] = np.mean(inp[i*r:(i+1)*r,j*r:(j+1)*r])
    return outp


# In[67]:

def blur(orig,a):
    A = np.dot(np.transpose(idm(lenna_dim**2,a)),idm(lenna_dim**2,a))
    return np.dot(A,orig)


# In[10]:

# prox operator
def prox(x):
    d = np.shape(x)
    y=x
    for j in range(0,d[0]):
        if x[j,0] > 1:
            y[j,0] = 1
            
        elif x[j,0] < -1:
            y[j,0] = -1
    return(y)


# In[55]:

# multi prox papc implementation for lenna tests
def papc(L,A,b,M,N,v0,u0,t,s,A2):
    
    # initialization
    v = v0
    u = u0
    u1 = u
    Mt = np.transpose(M)
    Ab = np.dot(np.transpose(A),b)
    U1 = []
    U = []


    # iterations
    for i in range(N):
        p = u
        for j in range(0,L):
            p = p-t*(np.dot(Mt,v) + 2*np.dot(A2,p) - 2*Ab )
            a = v + s*np.dot(M,p)
            v = prox(a)
        u = u-t*(np.dot(Mt,v) + 2*np.dot(A2,u) - 2*Ab )
        u1 = (u + i*u1)/(i+1)
        #V.append(v)
        U.append(u)
        U1.append(u1)
        
    return{'U1':U1,'U':U,'optval':None}


# In[12]:

# PAPC preliminary
import time as time
t0 = time.time()
u0 = np.zeros(lenna_n.shape)
v0 = np.dot(M,u0)
A2 = np.dot(np.transpose(A),A)
t = 1/(2*np.linalg.norm(A2,2))
print('t:',t)
if t > 1:
    t = np.random.rand(1)
    print('t:',t)
t1 = time.time()
print('time:',t1 - t0)
s = t*(np.linalg.norm(M,2))**2
t2 = time.time()
print('\ns:',s,'\ntime:',t2-t1)


# In[62]:

# actual PAPC run
N = 200
t0 = time.time()
R = papc(1,A,lenna_b,M,N,v0,u0,t,s,A2)
print('papc\ntime:',time.time()-t0,'\n')


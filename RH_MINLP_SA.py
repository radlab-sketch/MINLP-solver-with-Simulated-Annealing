# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
# This program provides the implementation of a receding horizon mixed integer non-linear programming (RH-MINLP) motion planner
# for n quadrotors. Simulated annealing is used to handle the non-convex constraints on communication connectivity between the quadrotors.
# This program will run in Python 3.7 or later.
# The number of quadrotors in simulation can be changed by modifying the variable 'nrobots'

#from numpy import concatenate, zeros, eye, tile, ones, multiply, divide
import numpy as np
from scipy import interpolate
import math
import random as rn
import time
np.set_printoptions(threshold=np.inf)
from scipy.sparse.linalg import inv
global nrobots
global thor
global etasize
global n_const
global tau
global nconn
global Str
global fq
global Noise
global sp
global absorb
global h
global eta_SNR
global M1
global M2
global gamma_a
global gamma_b
global A_0
global cu_in
global prev_in
global comm_range
global utminus1
global deltat
global smin
global smax
global stminus1
global amin
global amax
global dsafe
global A
global parameter
global megadecisionvalues
global currentrobotnumber
global symb
global currentglobaltime
global iterlim
global SNRapp
global u_bound_A1
global l_bound_A1
global u_bound_A21
global l_bound_A21
global u_bound_A22
global l_bound_A22
global u_bound_A3
global l_bound_A3
global u_bound_A4
global l_bound_A4
global u_bound_A5
global l_bound_A5
global J0
global rhs_A1
global u0
global s0


nrobots = 5
thor = 1
etasize = 5*thor
n_const = 4*nrobots+11
tau = 10
nconn = 1
Str = 30
fq = 15
Noise = (10**(5-1.8*math.log10(fq)))*(10**(-6))
sp = 1.5
absorb = 10**(0.001*((0.011*((fq**2)/(1+fq**2)))+(4.4*((fq**2)/(4100+fq**2)))+(2.75*10**(-5)*fq**2)+0.0003))
h = 20
eta_SNR = 10**(24/10)
M1 = 2
M2 = 2
gamma_a = 0.0193
gamma_b = 1
A_0 = 15
cu_in = np.ones((thor,nrobots))
prev_in = np.ones((thor,nrobots))
comm_range = 3
utminus1 = 0
delta_t = 0.5
smin = 0
smax = 2
stminus1 = 0
amin = -1
amax = 0.5
dsafe = 0.02
A = 100
iterlim = 20
SNRapp = 1000 # set to this value in previous code

#if deltat == 1:
    #gt = 19
    #if deltat == 0.5:
        #gt = 20
        #if deltat == 0.25:
            #gt = 40
            #if deltat == 0.1:
                #gt = 80


# In[2]:



def eta_gen(waypoints,r1,c):

    global arclength
    global xsplcoeffs
    global ysplcoeffs
    global zsplcoeffs
    global finalsymeta
    global eta

    # waypoints = np.concatenate((waypoint1,waypoint2,waypoint3,waypoint4,waypoint5),axis=0)

    arclength = np.zeros((nrobots,max(r1)))
    #arclength=[]
    cubsplx = np.zeros((nrobots,1),dtype=object)
    cubsply = np.zeros((nrobots,1),dtype=object)
    cubsplz = np.zeros((nrobots,1),dtype=object)
    xsplcoeffs = np.zeros((nrobots*4,4)) # to store spline coefficients
    ysplcoeffs = np.zeros((nrobots*4,4))
    zsplcoeffs = np.zeros((nrobots*4,4))
    finalsymeta = np.zeros((etasize,nrobots)) # creates an array where each column stores decision variables for the robots
    # to check for x, y, and z components of waypoints
    if c == 3:
        for j in robots:
            # arclength_elem=np.zeros((r1[j-1]))
            # goes from row to row

            for k in range(1,r1[j-1]):
                # arclength is ...?


                arclength[j-1,k] = arclength[j-1,k-1] + np.sqrt((waypoints[sum(r1[0:j-1])+k,0]-waypoints[sum(r1[0:j-1])+k-1,0])**2 + (waypoints[sum(r1[0:j-1])+k,1]-waypoints[sum(r1[0:j-1])+k-1,1])**2 + (waypoints[sum(r1[0:j-1])+k,2]-waypoints[sum(r1[0:j-1])+k-1,2])**2)
                # arclength_elem[k] = arclength_elem[k-1] + np.sqrt((waypoints[sum(r1[0:j-1])+k,0]-waypoints[sum(r1[0:j-1])+k-1,0])**2 + (waypoints[sum(r1[0:j-1])+k,1]-waypoints[sum(r1[0:j-1])+k-1,1])**2 + (waypoints[sum(r1[0:j-1])+k,2]-waypoints[sum(r1[0:j-1])+k-1,2])**2)
            #arclength.append(arclength_elem)
        for i in range(1,thor+1):
            for j in robots:
                cubsplx[j-1] = interpolate.CubicSpline(arclength[j-1][0:r1[j-1]],waypoints[sum(r1[0:j-1]):sum(r1[0:j]),0])
                cubsply[j-1] = interpolate.CubicSpline(arclength[j-1][0:r1[j-1]],waypoints[sum(r1[0:j-1]):sum(r1[0:j]),1])
                cubsplz[j-1] = interpolate.CubicSpline(arclength[j-1][0:r1[j-1]],waypoints[sum(r1[0:j-1]):sum(r1[0:j]),2])
                xsplcoeffs[(j-1)*(r1[j-1]-1):j*(r1[j-1]-1),:] = np.transpose(cubsplx[j-1,0].c)
                ysplcoeffs[(j-1)*(r1[j-1]-1):j*(r1[j-1]-1),:] = np.transpose(cubsply[j-1,0].c)
                zsplcoeffs[(j-1)*(r1[j-1]-1):j*(r1[j-1]-1),:] = np.transpose(cubsplz[j-1,0].c)

                finalsymeta[i-1,j-1] = cubsplx[j-1,0](i-1)
                finalsymeta[i+thor-1,j-1] = cubsply[j-1,0](i-1)
                finalsymeta[i+2*thor-1,j-1] = cubsplz[j-1,0](i-1)
                finalsymeta[i+3*thor-1,j-1] = 0.1 # initialize to prevent zero in denominator
                finalsymeta[i+4*thor-1,j-1] = 0.1 # initialize to prevent zero in denominator
                finalsymeta[i+5*thor-1:,j-1] = 0.9

        finemesh=np.array([np.linspace(0,arclength[0][-1],100)])
    return finalsymeta,cubsplx,cubsply,cubsplz,finemesh


# In[3]:


def spline_gen(rbtnum,eta):
    
    global xsymcoeffs
    global ysymcoeffs
    global zsymcoeffs
    global arclength_break
    global U
    
    ut = eta[3*thor:4*thor]
    xsymcoeffs = np.zeros((4,thor))
    ysymcoeffs = np.zeros((4,thor))
    zsymcoeffs = np.zeros((4,thor))
    arclength_break = np.zeros((thor,1))
    
    for i in range(1,len(ut)+1):
        if arclength[rbtnum-1,0] <= ut[i-1] < arclength[rbtnum-1,1]:
            cu_in[i-1,rbtnum-1] = (rbtnum-1)*4
            r = 0
        elif arclength[rbtnum-1,1] <= ut[i-1] < arclength[rbtnum-1,2]:
            cu_in[i-1,rbtnum-1] = (rbtnum-1)*4+1
            r = 1
        elif arclength[rbtnum-1,2] <= ut[i-1] < arclength[rbtnum-1,3]:
            cu_in[i-1,rbtnum-1] = (rbtnum-1)*4+2
            r = 2
        elif arclength[rbtnum-1,3] <= ut[i-1] <= arclength[rbtnum-1,4]:
            cu_in[i-1,rbtnum-1] = (rbtnum-1)*4+3
            r = 3
        elif ut[i-1] > arclength[rbtnum-1,4]:
            cu_in[i-1,rbtnum-1] = (rbtnum-1)*4+3
            r = 3
        elif ut[i-1] <= arclength[rbtnum-1,0]:
            cu_in[i-1,rbtnum-1] = (rbtnum-1)*4
            r = 4
            
        prev_in[i-1,rbtnum-1] = cu_in[i-1,rbtnum-1]

        xsymcoeffs[:,i-1] = xsplcoeffs[int(cu_in[i-1,rbtnum-1])] # might need to change when data is generated
        ysymcoeffs[:,i-1] = ysplcoeffs[int(cu_in[i-1,rbtnum-1])] # might need to change when data is generated
        zsymcoeffs[:,i-1] = zsplcoeffs[int(cu_in[i-1,rbtnum-1])] # might need to change when data is generated
        arclength_break[i-1,0] = arclength[rbtnum-1,r] # int(cu_in[i-1,rbtnum-1])

    U = arclength[rbtnum-1,-1]
    
    return xsymcoeffs, ysymcoeffs, zsymcoeffs, arclength_break, U


# In[4]:


def sym_obj_fun(eta,U):
    
    objective = U*thor - sum(eta[3*thor:4*thor]) # should work with integers
    
    return objective

def obj_fun_grad(eta):
    
    gradient = np.zeros((etasize,1))
    gradient[3*thor:4*thor] = -1
    
    return gradient


# In[5]:


def constraints0(rbtnum):
    
    global u_bound_A1
    global l_bound_A1
    global u_bound_A21
    global l_bound_A21
    global u_bound_A3
    global l_bound_A3
    global u_bound_A4
    global l_bound_A4
    global u_bound_A5
    global l_bound_A5
    global J0
    global rhs_A1
    global u0
    global s0
    
    # A1 --> linear equality constraints
    A1 = np.concatenate((np.zeros((thor, 3*thor)), np.eye(thor) - np.eye(thor, k = -1), np.eye(thor) * -delta_t), axis = 1)

    rhs_A1 = np.zeros((thor, 1))
    rhs_A1[0] = u0[rbtnum-1]

    # A2 --> linear inequality constraints
    A2 = np.concatenate((np.zeros((thor, 4*thor)), np.eye(thor) - np.eye(thor, k = -1)), axis = 1)

    # creating the bounds to go with this matrix

    # b
    l_bound_A21 = np.ones((thor, 1)) * (amin * delta_t)
    l_bound_A21[0] = l_bound_A21[0] + s0[rbtnum-1]

    # b + r
    u_bound_A21 = np.ones((thor, 1)) * (amax * delta_t)
    u_bound_A21[0] = u_bound_A21[0] + s0[rbtnum-1]

    # A3 --> nonlinear equality constraints (spline or path following constraints)
    A3 = np.concatenate((np.eye(3*thor), np.tile(np.eye(thor), (3, 1)), np.zeros((3*thor, thor))), axis = 1)
    A30 = np.concatenate((np.eye(3*thor), np.zeros((3*thor, 2*thor))), axis = 1)

    rhs_A3 = np.zeros((3*thor, 1))

    # A4 --> nonlinear inequality constraints (collision avoidance constraints)
    A4 = np.concatenate((np.tile(np.eye(thor), ((nrobots - 1), 3)), np.zeros(((nrobots - 1)*thor, 2*thor))), axis = 1)
    A40 = np.zeros(((nrobots-1)*thor,5*thor))

    l_bound_A4 = np.ones(((nrobots-1)*thor,1)) * dsafe
    u_bound_A4 = np.ones(((nrobots-1)*thor,1)) * float("inf")

    # A5 --> nonlinear inequality constraints (communication constraints)
    A5 = np.concatenate((np.tile(np.kron(np.eye(thor),np.ones((nconn,1))), (1,3)), np.zeros((nconn*thor, 2*thor))), axis = 1)
    A50 = np.zeros((nconn*thor, 5*thor))
    
    l_bound_A5 = np.ones((nconn*thor,1)) * -SNRapp
    u_bound_A5 = np.ones((nconn*thor,1)) * float("inf")

    J0 = csc_matrix(np.concatenate((A5, A4, A3, A2, A1))) # to promote easier indexing for Jacobian
    con = csc_matrix(np.concatenate((A50, A40, A30, A2, A1))) # to promote easier indexing for Jacobian
    
    return con


# In[6]:


def constraints(rbtnum,con,eta,comm_bin):
    
    con_val = np.zeros((thor*(nrobots + nconn + 4),1))
    others = np.ones(nrobots,dtype=bool)
    others[rbtnum-1] = False
    
    for i in range(0, thor):
        # create temporary variable for indices of robots in communication; needed to convert to integers
        temp_bin = comm_bin[:,i].astype(int)
        
        # does creating additional variables slow down the code?
        x_other = finalsymeta[i,others]
        y_other = finalsymeta[i+thor,others]
        z_other = finalsymeta[i+2*thor,others]
        
        # d_ij_t
        con_val[i:nconn*thor:thor] = -np.reshape((eta[i] - x_other[temp_bin])**2 + (eta[i+thor] - y_other[temp_bin])**2 + (eta[i+2*thor] - z_other[temp_bin])**2, (nconn,1))
        con_val[nconn*thor+i:thor*(nrobots+nconn-1):thor] = np.reshape((eta[i] - finalsymeta[i,others])**2 + (eta[i+thor] - finalsymeta[i+thor,others])**2 + (eta[i+2*thor] - finalsymeta[i+2*thor,others])**2, ((nrobots-1),1))
        
        # x, y, z positions on spline
        con_val[thor*(nrobots+nconn-1)+i] = -(xsymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**3 + xsymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i])**2 + xsymcoeffs[2,i]*(eta[3*thor+i]-arclength_break[i]) + xsymcoeffs[3,i])
        con_val[thor*(nrobots+nconn)+i] = -(ysymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**3 + ysymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i])**2 + ysymcoeffs[2,i]*(eta[3*thor+i]-arclength_break[i]) + ysymcoeffs[3,i])
        con_val[thor*(nrobots+nconn+1)+i] = -(zsymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**3 + zsymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i])**2 + zsymcoeffs[2,i]*(eta[3*thor+i]-arclength_break[i]) + zsymcoeffs[3,i])
    
    con_val = con.dot(eta) + con_val # also could potentially use the @ operator
    
    return con_val


# In[7]:


def jacobian0(rbtnum,con):
    
    J = J0
    others = np.ones(nrobots,dtype=bool)
    others[rbtnum-1] = False
    
    for i in range(0, thor):
        # create temporary variable for indices of robots in communication; needed to convert to integers
        temp_bin = comm_bin[:,i].astype(int)
        
        
        x_other = finalsymeta[i,others]
        y_other = finalsymeta[i+thor,others]
        z_other = finalsymeta[i+2*thor,others]

        # for x
        J.data[J.indptr[i]:J.indptr[i]+nconn] = -2*(eta[i] - x_other[temp_bin])
        J.data[J.indptr[i]+nconn:J.indptr[i]+nrobots+nconn-1] = 2*(eta[i] - finalsymeta[i,others])

        # for y
        J.data[J.indptr[i+thor]:J.indptr[i+thor]+nconn] = -2*(eta[i+thor] - y_other[temp_bin])
        J.data[J.indptr[i+thor]+nconn:J.indptr[i+thor]+nrobots+nconn-1] = 2*(eta[i+thor] - finalsymeta[i+thor,others])

        # for z
        J.data[J.indptr[i+2*thor]:J.indptr[i+2*thor]+nconn] = -2*(eta[i+2*thor] - z_other[temp_bin])
        J.data[J.indptr[i+2*thor]+nconn:J.indptr[i+2*thor]+nrobots+nconn-1] = 2*(eta[i+2*thor] - finalsymeta[i+2*thor,others])

        # for u
        J.data[J.indptr[3*thor+i]:J.indptr[3*thor+i+1]][0:3] = [-3*xsymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**2 - 2*xsymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i]) - xsymcoeffs[2,i], -3*ysymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**2 - 2*ysymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i]) - ysymcoeffs[2,i], -3*zsymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**2 - 2*zsymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i]) - zsymcoeffs[2,i]]

    return J


# In[8]:


def jacobian(rbtnum,con,J):
    
    others = np.ones(nrobots,dtype=bool)
    others[rbtnum-1] = False
    
    for i in range(0, thor):
        # create temporary variable for indices of robots in communication; needed to convert to integers
        temp_bin = comm_bin[:,i].astype(int)
        
        
        x_other = finalsymeta[i,others]
        y_other = finalsymeta[i+thor,others]
        z_other = finalsymeta[i+2*thor,others]

        # for x
        J.data[J.indptr[i]:J.indptr[i]+nconn] = -2*(eta[i] - x_other[temp_bin])
        J.data[J.indptr[i]+nconn:J.indptr[i]+nrobots+nconn-1] = 2*(eta[i] - finalsymeta[i,others])

        # for y
        J.data[J.indptr[i+thor]:J.indptr[i+thor]+nconn] = -2*(eta[i+thor] - y_other[temp_bin])
        J.data[J.indptr[i+thor]+nconn:J.indptr[i+thor]+nrobots+nconn-1] = 2*(eta[i+thor] - finalsymeta[i+thor,others])

        # for z
        J.data[J.indptr[i+2*thor]:J.indptr[i+2*thor]+nconn] = -2*(eta[i+2*thor] - z_other[temp_bin])
        J.data[J.indptr[i+2*thor]+nconn:J.indptr[i+2*thor]+nrobots+nconn-1] = 2*(eta[i+2*thor] - finalsymeta[i+2*thor,others])

        # for u
        J.data[J.indptr[3*thor+i]:J.indptr[3*thor+i+1]][0:3] = [-3*xsymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**2 - 2*xsymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i]) - xsymcoeffs[2,i], -3*ysymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**2 - 2*ysymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i]) - ysymcoeffs[2,i], -3*zsymcoeffs[0,i]*(eta[3*thor+i]-arclength_break[i])**2 - 2*zsymcoeffs[1,i]*(eta[3*thor+i]-arclength_break[i]) - zsymcoeffs[2,i]]

    return J


# In[9]:


def hessian(rbtnum,H,lag,Dn):
    
    
    for i in range(0, thor):
        H.data[i:3*thor:thor] = 0.5/(sum(lag[i:nconn*thor:thor])-sum(lag[nconn*thor+i:(nconn+nrobots-1)*thor:thor]))
        
    der2_psx = np.multiply(np.reshape(6*xsymcoeffs[0,0:thor],(-1,1)),(eta[3*thor:4*thor]-arclength_break[0:thor])) + np.reshape(2*xsymcoeffs[1,0:thor],(-1,1))
    der2_psy = np.multiply(np.reshape(6*ysymcoeffs[0,0:thor],(-1,1)),(eta[3*thor:4*thor]-arclength_break[0:thor])) + np.reshape(2*ysymcoeffs[1,0:thor],(-1,1))
    der2_psz = np.multiply(np.reshape(6*zsymcoeffs[0,0:thor],(-1,1)),(eta[3*thor:4*thor]-arclength_break[0:thor])) + np.reshape(2*zsymcoeffs[1,0:thor],(-1,1))
 
    lag_xspl = lag[thor*(nrobots+nconn-1):thor*(nrobots+nconn)]
    lag_yspl = lag[thor*(nrobots+nconn):thor*(nrobots+nconn+1)]
    lag_zspl = lag[thor*(nrobots+nconn+1):thor*(nrobots+nconn+2)]

    H.data[3*thor:4*thor] = np.reshape(1/(np.multiply(der2_psx,lag_xspl) + np.multiply(der2_psy,lag_yspl) + np.multiply(der2_psz,lag_zspl) + diag_term[:thor,]),(1,thor))
    H.data[4*thor:] = np.reshape(1/diag_term[thor:],(1,thor))
    
    return H


# In[10]:


time00 = time.time()

from scipy.sparse import csc_matrix, hstack, vstack
from scipy.sparse.linalg import inv
#import sksparse
# from sksparse.cholmod import analyze, cholesky
import warnings
warnings.simplefilter('default', RuntimeWarning)

global robots
global iter
global u0
global s0

robots = list(range(1,nrobots+1)) # creats a list from 1 to the total # of robots
#robots = rn.sample(robots,len(robots))
init_rbtnum = robots[0]

iter = 0

time1 = 0
time2 = 0
time3 = 0
time4 = 0

# random waypoints that are subject to change between trials
waypoint1 = np.transpose([[-2.69451755502415, -2.55646618467774, -1.09097779954495, 0.482278865826261, 0.776059913150126],
                      [2.26848700711186, 0.731410216589463, -0.317215517557715, 0.102873589277452, 0.688151235999186],
                      [6.71364156752760, 7.70665686101656, 8.05374543277496, 8.89999079973634, 10.6143085609467]])
waypoint2 = np.transpose([[-2.07328311997206, -0.849105726033831, -0.952365346881993, -1.16339324796858, -0.0949455310078110],
                  [0.659011298863126, 1.35966000196410, 1.57185764868985, 0.315340454433260, -1.10636527970990],
                  [5.70188284072424, 6.87586057344904, 8.69576502790440, 10.0165126871568, 10.4691879521602]])
waypoint3 = waypoint1 + 1.3475
waypoint4 = waypoint2 - 0.7849
waypoint5 = waypoint2 + 0.6389
waypoints = np.concatenate((waypoint1, waypoint2, waypoint3, waypoint4, waypoint5),axis=0)

gt = 11
(r,c) = np.shape(waypoint1)

l = []
r = [r] * nrobots
finalsymeta,cubsplx,cubsply,cubsplz,finemesh=eta_gen(waypoints, r, c)
finalmegasymeta = np.zeros((gt, etasize, nrobots))
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')

print('************')

u0 = np.zeros((nrobots))
s0 = np.zeros((nrobots))

for j in robots:
    f1=cubsplx[j-1,:][0]
    f2=cubsply[j-1,:][0]
    f3=cubsplz[j-1,:][0]
    ax.plot3D(f1(finemesh)[0],f2(finemesh)[0],f3(finemesh)[0])    

def controller(u0,s0,r,c):
    global robots, comm_bin,eta,diag_term
    time0 = time.time()
    finalsymeta, cubsplx, cubsply, cubsplz, finemesh= eta_gen(waypoints, r, c)
    #print(time.time() - time0)
    #print(finalsymeta)
    eta = np.reshape(finalsymeta[:,init_rbtnum-1],(etasize,1))

    U = spline_gen(init_rbtnum,eta)[4]

    for rbtnum in robots:

        best_obj = float("inf")

        # generate random incumbent
        comm_bin = np.zeros((nconn, thor))
        neighbor_x = np.zeros((nconn, thor))
        neighbor_y = np.zeros((nconn, thor))
        neighbor_z = np.zeros((nconn, thor))

        for i in range(0,thor):
            temp_bin = np.random.choice(nrobots-1, nconn, replace=False)
            if i==0:
                temp_bin = 0
            else:
                temp_bin = 2
            x_other = finalsymeta[i,:]
            y_other = finalsymeta[i+thor,:]
            z_other = finalsymeta[i+2*thor,:]
            neighbor_x[:,i] = x_other[temp_bin]
            neighbor_y[:,i] = y_other[temp_bin]
            neighbor_z[:,i] = z_other[temp_bin]
            comm_bin[:,i] = temp_bin

        incumbent = comm_bin 
        l = np.zeros((2*thor,1))
        u = np.zeros((2*thor,1))
        u[0:thor] = U
        l[thor:2*thor] = smin
        u[thor:2*thor] = smax
        H = csc_matrix(np.eye(etasize))

        # set initial temperature, cooling ratio, and loop length for simulated annealing
        r = 0.9
        L = 3 
        T = 10
        T2 = ((nrobots-1)**thor)/L
        T = 0.01/(r**T2)

        #print(time.time()-time00)
        #T = 0.01002
        while T > 0.01:
            for i in range(0,L):
                #print(comm_bin)
                iter = 0
                mu = 10

                eta = np.reshape(finalsymeta[:,rbtnum-1],(etasize,1))

                
                lag = np.random.random_sample((thor*(nrobots + nconn + 4),1)) + 1
                lag[:nconn*thor] = mu/SNRapp
                lag[nconn*thor:(nconn+nrobots-1)*thor] = mu/SNRapp;
                #lag[(nconn+nrobots-1)*thor:(nconn+nrobots+2)*thor] = 3
                lag[(nconn+nrobots+2)*thor:(nconn+nrobots+3)*thor] = 4
                lag[(nconn+nrobots+3)*thor:(nconn+nrobots+4)*thor] = 5
                w = np.ones((thor*(nrobots + nconn + 4),1))  # initialize slack/surplus variables to a strictly positive value (1)
                w[:nconn*thor] = SNRapp
                w[nconn*thor:(nconn+nrobots-1)*thor] = SNRapp/2
                w[thor*(nrobots + nconn - 1):thor*(nrobots + nconn + 2)] = 0.0
                w[thor*(nrobots + nconn + 3):] = 0.0

                lagl = np.zeros((2*thor,1)) + 1
                lagu = np.zeros((2*thor,1)) + 1
                q = np.zeros((thor,1)) + 1 # initialize lagrange multipliers for two-sided constraints (equality)
                p = np.zeros((thor,1)) + 1 # initialize slack variables for two-sided constraints

                H.data[4*thor:] = 0
                con_mat = constraints0(rbtnum)
                J = jacobian0(rbtnum,con_mat)
                e = np.zeros((thor*(nrobots + nconn + 4),1))
                rhs2 = np.zeros((thor*(nrobots + nconn + 4),1))

                while mu > 1.0e-08 and iter < 25:

                    x_free = eta[:3*thor]
                    x_bdd = eta[3*thor:]

                    w_A5 = w[:nconn*thor]
                    w_A4 = w[nconn*thor:thor*(nrobots+nconn-1)]
                    w_A3 = w[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)]
                    w_A21 = w[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]
                    w_A1 = w[thor*(nrobots+nconn+3):]

                    y_A5 = lag[:nconn*thor]
                    y_A4 = lag[nconn*thor:thor*(nrobots+nconn-1)]
                    y_A3 = lag[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)]
                    y_A21 = lag[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]
                    y_A1 = lag[thor*(nrobots+nconn+3):]

                    xsymcoeffs, ysymcoeffs, zsymcoeffs, arclength_break, U = spline_gen(rbtnum,eta)

                    # initialize objective value
                    prev_objective = sym_obj_fun(eta,U)

                    # initialize constraint values; see if con_mat*eta is better inside or outside the function
                    prev_con = constraints(rbtnum,con_mat,eta,comm_bin) # ...,eta,xsymcoeffs,ysymcoeffs,zsymcoeffs,arclength_break,U) # rbtnum,eta,U

                    inf_A5 = l_bound_A5 - prev_con[:nconn*thor]
                    inf_A4 = l_bound_A4 - prev_con[nconn*thor:thor*(nrobots+nconn-1)]
                    inf_A3 = prev_con[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)]
                    inf_A21l = l_bound_A21 - prev_con[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]
                    inf_A21u = prev_con[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)] - (u_bound_A21+l_bound_A21)
                    inf_A1 = prev_con[thor*(nrobots+nconn+3):] - rhs_A1
                    prev_inf = sum(np.maximum(inf_A5,np.zeros(inf_A5.shape))) + sum(np.maximum(inf_A4,np.zeros(inf_A4.shape))) + sum(abs(inf_A3)) + sum(np.maximum(inf_A21l,np.zeros(inf_A21l.shape))) + sum(np.maximum(inf_A21u,np.zeros(inf_A21u.shape))) + sum(abs(inf_A1))

                    # initialize Jacobian of constraints matrix
                    # time0 = time.time()
                    J = jacobian(rbtnum,con_mat,J)
                    # time1 = time1 + time.time() - time0

                    # initialize gradient of Lagrangian; should I be using this equation or the obj_fun_grad() function?
                    gradient = obj_fun_grad(eta) - J.T.dot(lag)

                    # initialize Hessian of constraints matrix
                    # time0 = time.time()
                    diag_term = np.divide(lagl,x_bdd-l) + np.divide(lagu,u-x_bdd)
                    iH = hessian(rbtnum,H,lag,diag_term) # should be positive definite matrix
                    # time2 = time2 + time.time() - time0
                    #print(time2)                

                    e[:nconn*thor] = np.divide(w_A5,y_A5)
                    e[nconn*thor:thor*(nrobots+nconn-1)] = np.divide(w_A4,y_A4)
                    e[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)] = 0.0
                    e[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)] = np.divide( np.ones((thor,1)), np.divide((y_A21+q),w_A21) + np.divide(q,p) )
                    e[thor*(nrobots+nconn+3):] = 0.0 
                    c = csc_matrix(np.diagflat(e)) # E

                    qdefmat = c + J*iH*J.T

                    rhs1 = gradient
                    rhs1[3*thor:] = rhs1[3*thor:] - mu/(x_bdd - l) + mu/(u - x_bdd)
                    rhs2[:nconn*thor] = l_bound_A5 - prev_con[:nconn*thor] + mu/y_A5
                    rhs2[nconn*thor:thor*(nrobots+nconn-1)] = l_bound_A4 - prev_con[nconn*thor:thor*(nrobots+nconn-1)] + mu/y_A4
                    rhs2[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)] = -prev_con[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)]
                    rhs2[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)] = l_bound_A21 - prev_con[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)] - e[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]*(mu/p - mu/w_A21 - q*(u_bound_A21-l_bound_A21)/p)
                    rhs2[thor*(nrobots+nconn+3):] = rhs_A1 - prev_con[thor*(nrobots+nconn+3):]
                    rhsmat = rhs2 + J*iH*rhs1

                    # time0 = time.time()
                    # if iter == 0:
                    #     SymFactor = analyze(qdefmat)
                    # Factor = SymFactor.cholesky(qdefmat)
                    # delta = Factor.solve_A(rhsmat)
                    #print(delta)


                    
                    # time0 = time.time()
                    qdefmat=qdefmat.todense()
                    delta = np.linalg.solve(qdefmat,rhsmat)
                    # time3 = time3 + time.time() - time0
                    # time3 = time3 + time.time() - time0

                    Dy_A5 = delta[:nconn*thor]
                    Dy_A4 = delta[nconn*thor:thor*(nrobots+nconn-1)]
                    Dy_A3 = delta[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)]
                    Dy_A21 = delta[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]
                    Dy_A1 = delta[thor*(nrobots+nconn+3):]

                    delta_eta = -iH*(rhs1 - J.T*delta)
                    Dx_free = delta_eta[:3*thor]
                    Dx_bdd = delta_eta[3*thor:]

                    Dw_A5 = mu/y_A5 - w_A5 - w_A5*Dy_A5/y_A5
                    Dw_A4 = mu/y_A4 - w_A4 - w_A4*Dy_A4/y_A4
                    Dw_A3 = np.zeros((3*thor,1))
                    Dw_A21 = -w_A21 - e[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]*(mu/p - mu/w_A21 - q*(u_bound_A21-l_bound_A21)/p + Dy_A21)
                    Dw_A1 = np.zeros((thor,1))

                    Dlagl = mu/(x_bdd-l) - lagl - lagl*Dx_bdd/(x_bdd-l)
                    Dlagu = mu/(u-x_bdd) - lagu + lagu*Dx_bdd/(u-x_bdd)

                    Dq = q/p*(Dw_A21-(u_bound_A21-l_bound_A21)+w_A21+mu/q)
                    Dp = (u_bound_A21-l_bound_A21)-w_A21-p-Dw_A21

                    # time0 = time.time()
                    all_list = list(np.concatenate((p,q,w_A5,w_A4,w_A21,lagl,lagu,y_A5,y_A4,y_A21+q,x_bdd-l,u-x_bdd),axis=0))
                    allD_list = list(np.concatenate((Dp,Dq,Dw_A5,Dw_A4,Dw_A21,Dlagl,Dlagu,Dy_A5,Dy_A4,Dy_A21+Dq,Dx_bdd,-Dx_bdd),axis=0))

                    # only care about where delta < 0
                    for i in reversed(range(0,thor*(2*nrobots +2*nconn + 10))):
                        if allD_list[i] >= 0:
                            del allD_list[i]
                            del all_list[i]

                    all_list = np.reshape(np.asarray(all_list),(-1,1)) # used to account for empty arrays w/ shape (1,)
                    allD_list = np.reshape(np.asarray(allD_list),(-1,1)) # used to account for empty arrays w/ shape (1,)
                    #print(np.concatenate((all_list,allD_list),axis=1))

                    alpha = -np.divide(all_list,allD_list)
                    alpha_max = np.zeros((alpha.shape)) + 1
                    alpha_bar = min(np.minimum(alpha, alpha_max)) # min with 1
                    if (alpha_bar < 1.0):
                        alpha_bar = 0.95*alpha_bar
                    #print(alpha_bar)
                    # time4 = time4 + time.time() - time0

                    comp_con = np.zeros((thor*(2*nrobots + 4),1))
                    new_eta = eta + alpha_bar*delta_eta
                    new_obj_val = sym_obj_fun(new_eta,U)
                    new_con = constraints(rbtnum,con_mat,new_eta,comm_bin)
                    inf_A5 = l_bound_A5 - new_con[:nconn*thor]
                    inf_A4 = l_bound_A4 - new_con[nconn*thor:thor*(nrobots+nconn-1)]
                    inf_A3 = new_con[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)]
                    inf_A21l = l_bound_A21 - new_con[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]
                    inf_A21u = new_con[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)] - (u_bound_A21+l_bound_A21)
                    inf_A1 = new_con[thor*(nrobots+nconn+3):] - rhs_A1
                    new_inf = sum(np.maximum(inf_A5,np.zeros(inf_A5.shape))) + sum(np.maximum(inf_A4,np.zeros(inf_A4.shape))) + sum(abs(inf_A3)) + sum(np.maximum(inf_A21l,np.zeros(inf_A21l.shape))) + sum(np.maximum(inf_A21u,np.zeros(inf_A21u.shape))) + sum(abs(inf_A1))

                    imp = 0

                    while ((new_obj_val >= prev_objective) and (new_inf > prev_inf) and (imp < 10)):
                        alpha_bar = 0.5*alpha_bar
                        new_eta = eta + alpha_bar*delta_eta
                        new_obj_val = sym_obj_fun(new_eta,U)
                        new_con = constraints(rbtnum,con_mat,new_eta,comm_bin)
                        inf_A5 = l_bound_A5 - new_con[:nconn*thor]
                        inf_A4 = l_bound_A4 - new_con[nconn*thor:thor*(nrobots+nconn-1)]
                        inf_A3 = new_con[thor*(nrobots+nconn-1):thor*(nrobots+nconn+2)]
                        inf_A21l = l_bound_A21 - new_con[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]
                        inf_A21u = new_con[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)] - (u_bound_A21+l_bound_A21)
                        inf_A1 = new_con[thor*(nrobots+nconn+3):] - rhs_A1
                        new_inf = sum(np.maximum(inf_A5,np.zeros(inf_A5.shape))) + sum(np.maximum(inf_A4,np.zeros(inf_A4.shape))) + sum(abs(inf_A3)) + sum(np.maximum(inf_A21l,np.zeros(inf_A21l.shape))) + sum(np.maximum(inf_A21u,np.zeros(inf_A21u.shape))) + sum(abs(inf_A1))
                        imp = imp + 1

                    alpha_star = alpha_bar

                    eta = eta + alpha_star*delta_eta 
                    lag = lag + alpha_star*delta
                    w = w + alpha_star*np.concatenate((Dw_A5,Dw_A4,Dw_A3,Dw_A21,Dw_A1),axis=0)
                    lagl = lagl + alpha_star*Dlagl
                    lagu = lagu + alpha_star*Dlagu
                    q = q + alpha_star*Dq
                    p = p + alpha_star*Dp
                    
                    oldmu = mu
                    mu = sum(w*lag)
                    mu = mu + sum(w[thor*(nrobots+nconn+2):thor*(nrobots+nconn+3)]*q)
                    mu = mu + sum((u-eta[3*thor:])*lagu)
                    mu = mu + sum((eta[3*thor:]-l)*lagl)
                    mu = mu + sum(p*q)
                    mu = 0.1*mu/((nrobots+nconn+6)*thor)
                    #print(mu)

                    iter += 1

             
                if new_obj_val <= best_obj:
                    best_obj = new_obj_val
                    best_eta = eta

                delta_obj = new_obj_val - prev_objective
                #print(delta_obj)

                if delta_obj <= 0:
                    incumbent = comm_bin
                else:
                    prob = math.exp(-delta_obj/T)
                    rand = np.random.random_sample()
                    if rand < prob:
                        incumbent = comm_bin

                #print(comm_bin)

                rand_row = np.random.randint(0,nconn)
                rand_col = np.random.randint(0,thor)
                comm_bin[rand_row,rand_col] = np.random.choice([x for x in range(0,nrobots-1) if x != comm_bin[rand_row,rand_col]])

                #print(comm_bin)
                #print(best_obj)

            T = r*T

        #print(iter)
        finalsymeta[:,rbtnum-1] = eta.flatten()
    return finalsymeta   

#In[11]:

# Set new values for u0 and s0 to obtain coordinates at each timestep
for i in range(0,gt):
    print('TimeStep : ',i)
    finalsymeta= controller(u0,s0,r,c)
    u0 = finalsymeta[3 * thor, :]
    s0 = finalsymeta[3 * thor + 1, :]
    finalmegasymeta[i,:,:] = finalsymeta


#In[12]:

def variables():
    return finalmegasymeta, gt, cubsplx, cubsply, cubsplz, finemesh, robots,waypoints, thor, nrobots


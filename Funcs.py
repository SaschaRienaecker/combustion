import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable
from poisson_solver import SOR_solver
from os import listdir
from numba import jit

def conv_x(tn:float, u:np.array,**kwargs):
    v  = kwargs['v']
    dx = kwargs['dx']
    dy = kwargs['dy']
    return (u * df1_2(u,dx,axis=0)+ v * df1_2(u,dy,axis=1))

def conv_y(tn:float, v:np.array,**kwargs):
    u  = kwargs['u']
    dx = kwargs['dx']
    dy = kwargs['dy']
    return (u * df1_2(v,dx,axis=0)+ v * df1_2(v,dy,axis=1))

def diff(tn:float, u:np.array, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    nu = kwargs['nu']
    return nu*(
      df2_2(u,dx,axis=0)+
      df2_2(u,dy,axis=1)
      )
def conv_x_diff(tn:float, u:np.array, **kwargs):
    v  = kwargs['v']
    dx = kwargs['dx']
    dy = kwargs['dy']
    nu = kwargs['nu']
    y= -(u * df1_2(u,dx,axis=0)+ v * df1_2(u,dy,axis=1))+nu*(
      df2_2(u,dx,axis=0)+
      df2_2(u,dy,axis=1)
      )
    return y
def conv_y_diff(tn:float, v:np.array, **kwargs):
    u  = kwargs['u']
    dx = kwargs['dx']
    dy = kwargs['dy']
    nu = kwargs['nu']
    return -(u * df1_2(v,dx,axis=0)+ v * df1_2(v,dy,axis=1))+nu*(
      df2_2(v,dx,axis=0)+
      df2_2(v,dy,axis=1)
      )

def u_st1(u:np.array,v:np.array,dx:float,dy:float,dt:float):
    u1 = RK33(conv_x,0,u ,dt ,v=v ,dx=dx,dy=dy)
    v1 = RK33(conv_y,0,v ,dt ,u=u ,dx=dx,dy=dy)
    return u1,v1

def u_st2(u:np.array,v:np.array,dx:float,dy:float,dt:float,nu:float):
    u1,v1 = u_st1(u,v,dx,dy,dt)
    u2 = RK33(diff  ,0 ,u1,dt,dx=dx, dy=dy, nu=nu)
    v2 = RK33(diff  ,0 ,v1,dt,dx=dx, dy=dy, nu=nu)
    return u2,v2

def u_dir(u:np.array,v:np.array,dx:float,dy:float,dt:float,nu:float):
    u = RK33(conv_x_diff,0,u ,dt ,v=v ,dx=dx,dy=dy,nu = nu)
    v = RK33(conv_y_diff,0,v ,dt ,u=u ,dx=dx,dy=dy,nu = nu)
    return [u,v]


def df1_1(data:np.array,dh:float,axis:int=0):
    # return (np.roll(data, 1, axis=axis)
    #         -np.roll(data, 0, axis=axis))/(dh)
    # without np.roll
    data_deriv = np.zeros_like(data)
    N = data.shape[0]
    M = data.shape[1]
    if axis == 0:
        N = data.shape[0]
        data_deriv[0:N-1,:] = (data[1:N,:] - data[0:N-1,:])/dh
    elif axis == 1:
        N = data.shape[1]
        data_deriv[:,0:N-1] = (data[:,0:N] - data[:,0:N-1])/dh
    return data_deriv

@jit(nopython=True)
def df1_2(data:np.array,dh:float,axis:int=0) :
    # return (np.roll(data, 1, axis=axis)
    #         -np.roll(data, -1, axis=axis))/(2*dh)
    data_deriv = np.zeros_like(data)
    if axis == 0:
        N = data.shape[0]
        data_deriv[1:N-1,:] = (data[2:N,:] - data[0:N-2,:])/(2*dh)
    elif axis == 1:
        N = data.shape[1]
        data_deriv[:,1:N-1] = (data[:,2:N] - data[:,0:N-2])/(2*dh)
    return data_deriv

def df1_4(data:np.array,dh:float,axis:int=0) :
        # return (np.roll(data,-2, axis=axis)
        #     - 8*np.roll(data,-1, axis=axis)
        #     + 8*np.roll(data, 1, axis=axis)
        #     -   np.roll(data, 2, axis=axis))/(12*dh)
        data_deriv = np.zeros_like(data)
        if axis == 0:
            N = data.shape[0]
            data_deriv[2:N-2,:] = (data[0:N-4,:] - 8 * data[1:N-3,:] + 8 * data[3:N-1,:] - data[4:N  ,:] )/(12*dh)
        elif axis == 1:
            N = data.shape[1]
            data_deriv[:,2:N-2] = (data[:,0:N-4] - 8 * data[:,1:N-3] + 8 * data[:,3:N-1] - data[:  ,4:N] )/(12*dh)
        return data_deriv

@jit(nopython=True)
def df2_2(data:np.array,dh:float,axis:int=0):
    # return (np.roll(data,1,axis=axis) - 2* np.roll(data,0,axis=axis) +np.roll(data,-1,axis=axis))/dh**2
    data_deriv = np.zeros_like(data)
    if axis == 0:
        N = data.shape[0]
        data_deriv[1:N-1,:] = (data[2:N,:] - 2 * data[1:N-1,:] + data[0:N-2  ,:])/(dh**2)
    elif axis == 1:
        N = data.shape[1]
        data_deriv[:,1:N-1] = (data[:,2:N] - 2 * data[:,1:N-1] + data[:,0:N-2])/(dh**2)
    return data_deriv



def RK11(func:callable,tn:float,yn:np.array,dh:float,**kwargs):
    return yn+ dh*func(tn     ,yn        ,**kwargs)

def RK33_T(func:callable,tn:float,yn:np.array,dh:float,H:float,**kwargs):
  res = [yn]
  for i in range(int(H/dh)):
    res.append(RK33(func,tn,res[-1],dh,**kwargs))
    tn+=dh
  return np.array(res)

def RK33(func:callable,tn:float,yn:np.array,dh:float,**kwargs):
    k1 = func(tn,yn              , **kwargs)
    k2 = func(tn,yn+dh*k1/2      , **kwargs)
    k3 = func(tn,yn-dh*k1+2*dh*k2, **kwargs)
    return  yn+ dh  *( k1/6+
                     2*k2/3+
                       k3/6)

def RK44(func:callable,tn:float,yn:np.array,dh:float,**kwargs):
    k1 = func(tn     ,yn        ,**kwargs)
    k2 = func(tn+dh/2,yn+dh*k1/2,**kwargs)
    k3 = func(tn+dh/2,yn+dh*k2/2,**kwargs)
    k4 = func(tn+dh  ,yn+dh*k3  ,**kwargs)
    return yn+ dh/6*(k1+
                    2*k2+
                    2*k3+
                    k4)
def means(data_cube):
    means =[]
    for i in range(len(data_cube)):
        means.append(np.std(data_cube[0]))
    return(means)




def save(data_cube,update_func,fig,file_type='avi',fargs=None):
    if True: #just for naming the gif file
        dirs = listdir('./')
        gifs = []
        for dir in dirs:
            gifs.append(dir) if dir[-3:]=='avi' else 0
            gifs.append(dir) if dir[-3:]=='gif' else 0

        f = r"./animation_"+"{:02d}.{}".format(len(gifs),file_type)
    if file_type == "gif":
        writer = animation.PillowWriter(fps=30)
    # Set up formatting for the movie files
    elif file_type=='avi':
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\hp\Documents\ffmpeg-2021-10-18-git-d04c005021-full_build\bin/ffmpeg.exe'
        writer = animation.writers['ffmpeg']
        writer = writer(fps=60, metadata=dict(artist='Slimane MZERGUAT'), bitrate=1800)

    # ani = animation.FuncAnimation(fig, update_func, interval=0.001, blit=True, frames=len(data_cube))#len(data_cube)-1)
    # plt.show()
    #for quiver
    ani = animation.FuncAnimation(fig, update_func, fargs=fargs, interval=0.1, blit=True, frames=len(data_cube[3]))#len(data_cube)-1)

    #ani.save(f, writer=writergif)
    print(f)
    ani.save(f, writer=writer)

def example():
    N       = 100
    M       = 100
    Lx      = 0.002
    Ly      = 0.002
    Lslot   = 0.0005
    Lcoflow = 0.0005
    dt= 10**-4
    nu =10**-3
    dx = Lx/N
    dy = Ly/N
    Ns_c = int(Lslot          /dx) #N for the point between the slot and te coflow
    Nc_lw= int((Lslot+Lcoflow)/dx) #N for the point between the coflow and lateral wall

    u =np.zeros((N+4,M+4))
    v =np.zeros((N+4,M+4))
    u[int((N+4)*1/4):int((N+4)*3/4),int((M+4)*1/4):int((M+4)*3/4)] = 1
    v[int((N+4)*1/4):int((N+4)*3/4),int((M+4)*1/4):int((M+4)*3/4)] = 1
    ures = [u]
    vres = [v]
    for i in range(1000):
        res = u_st2(ures[-1],vres[-1],dx,dy,dt,nu)
        ures.append(res[0])
        vres.append(res[1])

    ures = np.array(ures)
    vres = np.array(vres)
    return ures,vres


#### Functions implemented by Sascha for the species transport (tested with numba) ####

@jit(nopython=True)
def RK3(p, dt, F, *args):
    k1 = F(p, *args)
    k2 = F(p + dt /2 * k1, *args)
    k3 = F(p - dt * k1 + 2 * dt * k2, *args)
    return p + dt * (k1/6 + 2*k2/3 + k3/6)

@jit(nopython=True)
def RK4(p, dt, F, *args):
    k1 = F(p, *args)
    k2 = F(p + dt /2 * k1, *args)
    k3 = F(p + dt /2 * k2, *args)
    k4 = F(p + dt * k3, *args)
    return p + dt * (k1/6 + k2/3 + k3/3 + k4/6)

@jit(nopython=True)
def adv_diff(p, u, v, dx, dy, nu):
    return -(u * df1_2(p,dx,axis=0)+ v * df1_2(p,dy,axis=1)) + nu*(
      df2_2(p,dx,axis=0)+
      df2_2(p,dy,axis=1)
      )

@jit(nopython=True)
def advance_adv_diff_RK3(p, dt, u, v, dx, dy, nu):
    args = (u, v, dx, dy, nu)
    return RK3(p, dt, adv_diff, *args)

@jit(nopython=True)
def advance_adv_diff_RK4(p, dt, u, v, dx, dy, nu):
    args = (u, v, dx, dy, nu)
    return RK4(p, dt, adv_diff, *args)


@jit(nopython=True)
def diffusion(p, dx, dy, nu):
    """ Purely diffusive case for testing purposes. """
    return nu*(
    df2_2(p,dx,axis=0)+
    df2_2(p,dy,axis=1)
    )

@jit(nopython=True)
def advance_diff_RK3(p, dt, u, v, dx, dy, nu):
    args = (dx, dy, nu)
    return RK3(p, dt, diffusion, *args)

def tanh_transition(x, y1, y2, a, b):
    from numpy import tanh
    """ Hyperbolic tangent transition between two values y1, y2
    at a point a, with width b.
    Args:
        x (float or array-like):
            coordinates on which to evaluate this function.
        y1, y2 (float): values at -/+ infinity
        a (float): transition point, i.e. y''(a) = 0
        b (float): determines how smooth the transition is: high b -> sharp transition
    Returns:
        y (float or array-like):
            the function evaluated at point(s) x
    """
    y = y1 + (y2 - y1) * 0.5 * (1 + tanh((x - a) / b))
    return y

def metric_L0(a, b):
    norm = np.abs(a).max()
    return np.abs(a-b).max() / norm

def metric_RMS(a, b):
    norm = np.abs(a).max()
    return np.sqrt(np.mean((a-b)**2)) / norm

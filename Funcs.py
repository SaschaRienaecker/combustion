import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable
from os import listdir

                     
def conv_x(tn:float, u:np.array,**kwargs):
    v  = kwargs['v'] 
    dx = kwargs['dx'] 
    dy = kwargs['dy'] 
    return -(u * df1_2(u,dx,axis=0)+ v * df1_2(u,dy,axis=1))

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

def u_st1(u:np.array,v:np.array,dx:float,dy:float,dt:float):
    u1 = RK33(conv_x,0,u ,dt ,v=v ,dx=dx,dy=dy)
    v1 = RK33(conv_y,0,v ,dt ,u=u ,dx=dx,dy=dy)
    return u1,v1

def u_st2(u:np.array,v:np.array,dx:float,dy:float,dt:float,nu:float):
    u1,v1 = u_st1(u,v,dx,dy,dt)
    u2 = RK33(diff  ,0 ,u1,dt,dx=dx, dy=dy, nu=nu)
    v2 = RK33(diff  ,0 ,v1,dt,dx=dx, dy=dy, nu=nu)
    return u2,v2
    




def df1_1(data:np.array,dh:float,axis:int=0):
    return (np.roll(data, 1, axis=axis) 
            -np.roll(data, 0, axis=axis))/(dh)
def df1_2(data:np.array,dh:float,axis:int=0) :
    return (np.roll(data, 1, axis=axis) 
            -np.roll(data, -1, axis=axis))/(2*dh)
def df1_4(data:np.array,dh:float,axis:int=0) :
        return (np.roll(data,-2, axis=axis) 
            - 8*np.roll(data,-1, axis=axis) 
            + 8*np.roll(data, 1, axis=axis) 
            -   np.roll(data, 2, axis=axis))/(12*dh)

def df2_2(data:np.array,dh:float,axis:int=0):
    return (np.roll(data,1,axis=axis) - 2* np.roll(data,0,axis=axis) +np.roll(data,-1,axis=axis))/dh**2



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




def save(data_cube,update_func,fig,Dt,dt,file_type='avi'):

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

    ani = animation.FuncAnimation(fig, update_func, interval=0.1, blit=True, frames=len(data_cube))#len(data_cube)-1)
    #ani.save(f, writer=writergif)
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




# def df1_1(data,dh, axis=0):
# # size = data.shape
#     # print(size)
#     # data1 = np.zeros(size) *np.NaN
#     # print(data1.shape)

#     # if axis == 0: 
#     #     data1[  0:size[0]-1] = (data[1  :size[0]] - data[  0:size[0]-1])/dh 
#     # elif axis == 1:
#     #     data1[:,0:size[1]-1] = (data[:,1:size[1]] - data[:,0:size[0]-1])/dh
#     # return data1
#     size = data.shape
#     if axis == 0: 
#         return (data[1  :size[0]] - data[  0:size[0]-1])/dh 
#     elif axis == 1:
#         return (data[:,1:size[1]] - data[:,0:size[1]-1])/dh    


# def df1_4(data,dh,axis=0) :
#     # return (np.roll(data,-2, axis=axis) 
#     #     - 8*np.roll(data,-1, axis=axis) 
#     #     + 8*np.roll(data, 1, axis=axis) 
#     #     -   np.roll(data, 2, axis=axis))/(12*dh)

#     size = data.shape
#     if axis == 0: 
#         return (data[0  :size[0]-4] 
#             -8* data[1  :size[0]-3]
#             +8* data[3  :size[0]-1]
#             -   data[4  :size[0]  ])/(12*dh)
     
#     elif axis == 1:
#         return (data[:,0  :size[1]-4] 
#             -8* data[:,1  :size[1]-3]
#             +8* data[:,3  :size[1]-1]
#             -   data[:,4  :size[1]  ])/(12*dh)

# # def df2_2(data,dh,axis=0):
# #     return (np.roll(data,1,axis=axis) - 2* np.roll(data,0,axis=axis) +np.roll(data,-1,axis=axis))/dh**2
# #     if axis == 0   
# #     return (data[0  :size[0]-4] 
# #             -8* data[1  :size[0]-3]
# #             +8* data[3  :size[0]-1]
# #             -   data[4  :size[0]  ])/(12*dh)
     
# #     elif axis == 1:
# #         return (data[:,0  :size[1]-4] 
# #             -8* data[:,1  :size[1]-3]
# #             +8* data[:,3  :size[1]-1]
# #             -   data[:,4  :size[1]  ])/(12*dh)




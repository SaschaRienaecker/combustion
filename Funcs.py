import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import listdir

print("importing My TP1 functions")
if True: #u and Phi functions
    def df1_4(data,dh,axis=0) :
        return (np.roll(data,-2, axis=axis) 
            - 8*np.roll(data,-1, axis=axis) 
            + 8*np.roll(data, 1, axis=axis) 
            -   np.roll(data, 2, axis=axis))/(12*dh)

    def df1_1(data,dh,axis=0) :
        return (np.roll(data, 1, axis=axis) 
               -np.roll(data, -1, axis=axis))/(2*dh)
    
    def RK44(func,tn,yn,dh,**kwargs):
        k1 = func(tn     ,yn        ,**kwargs)  
        k2 = func(tn+dh/2,yn+dh*k1/2,**kwargs)
        k3 = func(tn+dh/2,yn+dh*k2/2,**kwargs)
        k4 = func(tn+dh  ,yn+dh*k3  ,**kwargs)
        return yn+ dh/6*(k1+
                     2*k2+
                     2*k3+
                     k4)
    def RK11(func,tn,yn,dh,**kwargs):  
        return yn+ dh*func(tn     ,yn        ,**kwargs)
    
    def df2_2(data,dh,axis=0):
        return (np.roll(data,1,axis=axis) - 2* np.roll(data,0,axis=axis) +np.roll(data,-1,axis=axis))/dh**2

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




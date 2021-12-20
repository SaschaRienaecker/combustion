#in this part i redefine advance_fluid_flow to calculate until stationarity

import matplotlib.pyplot as plt
import numpy as np 
from Funcs import advance_adv_diff_RK3
from fluid_flow import set_boundary,compute_P,df1_2
from numba import jit
import pickle
from parameters import *
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


u0 = np.zeros((N,M))
v0 = np.copy(u0)
u0,v0 = set_boundary(u0,v0)

u, v = np.copy(u0), np.copy(v0)
dt = 1e-7
# @jit(nopython=True)
def stdclose(A,N,rtot=0.01,std_extend=100,return_val = False, axis= 0):
    res = np.std(A[N-std_extend :N],axis = axis)/np.abs(np.mean(A[N-std_extend :N],axis=axis))
    res = np.nan_to_num(res,nan=0)
    if return_val:
        return np.all(res<=rtot),res
    else:
        return np.all(res<=rtot)

@jit(nopython=True)
def allclose(a,b,rtot=10**-5,atot=0):
    """    
    NOTE
    allclose doesn"t work with numba so i defined mine based on the documentation
    allcolose(a, b)=True if:
    |a - b| <= (  
                (rtot=st_ratio) * (|a|+|b|)/2 
               + atot=0 we don't care about this)"""
    
    return np.all(np.abs(a-b)<=atot+rtot*(np.abs(b)+np.abs(a))/2)


def advance_fluid_flow(st_ratio:float, u, v, f, dt):
    """
    st_ratio=ratio for stationnarity consideration 0< st_ratoio <1
    f: Time integration function, e.g. advance_adv_diff_RK3
    
    """
    print('evolving velocities')
    P = np.zeros_like(u)
    stationary = False
    # we keep data about the evolution for later
    U,V = np.zeros((2,10000,*u.shape)) 
    u,v = set_boundary(u,v)
    U[0] = u
    V[0] = v
    inc_N = 1
    while not stationary:
        u = f(U[inc_N-1], dt, U[inc_N-1], V[inc_N-1], dx, dy, nu)
        v = f(V[inc_N-1], dt, U[inc_N-1], V[inc_N-1], dx, dy, nu)
        u,v = set_boundary(u,v)
        # NOTE: using Pprev here increases Poisson solver convergence speed a lot!
        P = compute_P(u, v, dx, dt, rho, Pprev=P)

        dPdx = df1_2(P, dx, axis=0)
        dPdy = df1_2(P, dx, axis=1)

        u = u - dt / rho * dPdx
        v = v - dt / rho * dPdy
        u,v = set_boundary(u,v)

        if U.shape[0]<=inc_N: #to avoid append that's costful
            new_U,new_V = np.zeros((2,10000+U.shape[0],*u.shape)) 
            new_U[:U.shape[0]] = U.copy()
            new_V[:V.shape[0]] = V.copy()
            U=new_U.copy()
            V=new_V.copy()
        
        if False:#inc_N%10000==0: #plot results every 10000 
            fig, ax = plt.subplots()
            _,std_U = stdclose(U,inc_N,return_val=True)
            _,std_V = stdclose(V,inc_N,return_val=True)
            print(np.max(std_U) , np.max(std_V))
            im =ax.imshow(
                (std_U+std_V).T
                )
            ax.invert_yaxis()
            ax.quiver(U[inc_N-1].T, V[inc_N-1].T)
            fig.colorbar(im, ax=ax)
            plt.show()
            plt.plot(np.mean(np.nan_to_num(
                np.abs((U[0:-1]-U[1:]))*2/np.abs(U[0:-1]+U[1:])
                ,nan=0),axis=(1,2))
                +
                np.mean(np.nan_to_num(
                np.abs((V[0:-1]-V[1:]))*2/np.abs(V[0:-1]+V[1:])
                ,nan=0),axis=(1,2)
                ))
            plt.yscale('log')
            plt.show()
        U[inc_N]=u.copy()
        V[inc_N]=v.copy()
        #this technique could be wrong because it's seen that the results fluctuate so it's better to see the standard deviation
        # stationary =    (allclose(u,U[inc_N-1],rtot=st_ratio) 
        #         and 
        #             allclose(v,V[inc_N-1],rtot=st_ratio))

        #checking the stationarity by std over an extend of time and see if the overale variation is steady
        std_extend = 100
        stationary = stdclose(U,inc_N,std_extend=std_extend,rtot=0.01) and stdclose(V,inc_N,std_extend=std_extend,rtot=0.01) if inc_N>std_extend else False

        print(inc_N) if inc_N%1000==0 else 0
        inc_N+=1
    return U[:inc_N], V[:inc_N]

if False: #generating stationary velocity field
    U, V = advance_fluid_flow(0.001, u, v, advance_adv_diff_RK3, dt)
    pickle.dump([U,V],open("U_V_field-std_0.01.p",'wb'))
    fig, ax = plt.subplots(figsize=(4,4))
    ax.quiver(U[-1].T, V[-1].T)
    ax.set_title('velocity field')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()
    print(U.shape)

#animation comes after
def update(frame):
    # ln[1].set_data(V[frame].T**2+U[frame].T**2)
    ln[0].set_UVC(U[frame*50].T, V[frame*50].T)
    ln[1].set_array(V[frame*50].T**2+U[frame*50].T**2)
    ln[2].set_text('${:05.1f}\\mu s$'.format(frame*50*0.1))
    return ln


if False: #Analysing the results and generating video
    #saving a video
    U,V = pickle.load(open("U_V_field-std_0.01.p","rb"))
    fig, ax = plt.subplots(figsize=(16,16))
    ln1  = ax.quiver(U[0].T   ,V[0].T   ,scale=40,zorder=2)
    ln2  = ax.pcolormesh(U[0].T**2+V[0].T**2,zorder=1,vmin=0,vmax=1.75,cmap='rainbow')
    txt = ax.text(25,25,'$0\\mu s$',zorder=3,fontsize = 32)
    ax.set_xticklabels(ax.get_xticks()/N)
    ax.set_yticklabels(ax.get_yticks()/N)
    cbar = fig.colorbar(ln2,ax=ax)
    ln = [ln1,ln2,txt]

    ax.set_title("energie and direction evolution \nof the velocity until stability$ \\mu s$",fontsize = 32)
    ax.set_ylabel('y-direction (mm)',fontsize = 32)
    ax.set_xlabel('x-direction (mm)',fontsize = 32)
    plt.rcParams['animation.ffmpeg_path'] = r"C:\ffmpeg-N-99920-g46e362b765-win64-gpl-shared-vulkan\bin\ffmpeg.exe"
    writer = animation.writers['ffmpeg']
    writer = writer(fps=60, metadata=dict(artist='Slimane MZERGUAT'), bitrate=1800)
    ani = FuncAnimation(fig, update, blit=True, frames=int(len(U)/50),interval=18,repeat=False)
    ani.save("res.mp4", writer=writer)
    plt.show()
if True:# fluctuation analysis
    U,V = pickle.load(open("U_V_field-std_0.01.p","rb"))
    plt.plot(np.arange(0,len(U)-1)*0.1,
        np.mean(np.nan_to_num(
            np.abs((U[0:-1]-U[1:]))*2/np.abs(U[0:-1]+U[1:]),nan=0),axis=(1,2))
        +
        np.mean(np.nan_to_num(
            np.abs((V[0:-1]-V[1:]))*2/np.abs(V[0:-1]+V[1:]),nan=0),axis=(1,2)
        ))

    plt.yscale('log')
    plt.xlabel("time $(\\mu s)$",fontsize = 32)
    plt.ylabel('fluctuation amount',fontsize = 32)
    plt.title('quantifying the variability of the grid with time\n$\\langle\\delta r\\rangle^{x,y} = \\langle\\frac{2\\cdot|v_i-v_{i+1}|}{|v_i|+|v_{i+1}|}\\rangle^{x,y}$',fontsize = 32)
    plt.show()
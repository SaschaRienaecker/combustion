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
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    U,V,Pr = np.zeros((3,10000,*u.shape)) 
    u,v = set_boundary(u,v)
    U[0] = u
    V[0] = v
    Pr[0] = P
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
            new_U,new_V,new_Pr = np.zeros((3,10000+U.shape[0],*u.shape)) 
            new_U[:U.shape[0]] = U.copy()
            new_V[:V.shape[0]] = V.copy()
            new_Pr[:Pr.shape[0]] = Pr.copy()
            U=new_U.copy()
            V=new_V.copy()
            Pr=new_Pr.copy()
        
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
        Pr[inc_N] = P.copy()
        #this technique could be wrong because it's seen that the results fluctuate so it's better to see the standard deviation
        # stationary =    (allclose(u,U[inc_N-1],rtot=st_ratio) 
        #         and 
        #             allclose(v,V[inc_N-1],rtot=st_ratio))

        #checking the stationarity by std over an extend of time and see if the overale variation is steady
        std_extend = 100
        stationary = stdclose(U,inc_N,std_extend=std_extend,rtot=0.01) and stdclose(V,inc_N,std_extend=std_extend,rtot=0.01) if inc_N>std_extend else False

        print(inc_N) if inc_N%1000==0 else 0
        inc_N+=1
    return U[:inc_N], V[:inc_N], Pr[:inc_N]

if False: #generating stationary velocity field
    U, V,Pr = advance_fluid_flow(0.001, u, v, advance_adv_diff_RK3, dt)
    pickle.dump([U,V,Pr],open("U_V_field-std_0.01.p",'wb'))
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
    N_leap=50
    ln[0].set_UVC(U[frame*N_leap].T, V[frame*N_leap].T)
    ln[1].set_array(V[frame*N_leap].T**2+U[frame*N_leap].T**2)
    ln[2].set_array(Pr[frame*N_leap].T)
    ln3.set_clim(vmin=0,vmax=np.max(Pr[frame*N_leap]))
    ln[3].set_text('${:05.1f}\\mu s$'.format(frame*N_leap*0.1))
    print(frame*N_leap) 
    #  cbar2 = fig.colorbar(ln3, cax=cax)
    return ln


if False: #Analysing the results and generating video
    #saving a video
    U,V,Pr = pickle.load(open("U_V_field-std_0.01.p","rb"))
    print(np.max(Pr[:1000]))
    fig, (ax,ax2) = plt.subplots(1,2,figsize=(32,16))
    ln1  = ax.quiver(U[0].T   ,V[0].T   ,scale=40,zorder=2)
    ln2  = ax.pcolormesh(U[0].T**2+V[0].T**2,zorder=1,vmin=0,vmax=1.75,cmap='rainbow')
    ln3  = ax2.pcolormesh(Pr[0].T           ,zorder=1,                 cmap='rainbow')
    txt = ax.text(25,25,'$0\\mu s$',zorder=3,fontsize = 32)
    ax .set_xticklabels(ax .get_xticks()/N)
    ax .set_yticklabels(ax .get_yticks()/N)
    ax2.set_xticklabels(ax2.get_xticks()/N)
    ax2.set_yticklabels(ax2.get_yticks()/N)
    
    cbar  = fig.colorbar(ln2,ax=ax )
    div = make_axes_locatable(ax2)
    cax = div.append_axes('right', '5%', '5%')
    cbar2 = fig.colorbar(ln3, cax=cax)
    
    
    ln = [ln1,ln2,ln3,txt]

    ax .set_title("energie and direction evolution \nof the velocity until stability$ \\mu s$",fontsize = 32)
    ax .set_ylabel('y-direction (mm)',fontsize = 32)
    ax .set_xlabel('x-direction (mm)',fontsize = 32)
    ax2.set_title("pressure evolution of the velocity\n until stability$ \\mu s$",fontsize = 32)
    ax2.set_ylabel('y-direction (mm)',fontsize = 32)
    ax2.set_xlabel('x-direction (mm)',fontsize = 32)
    plt.rcParams['animation.ffmpeg_path'] = r"C:\ffmpeg-N-99920-g46e362b765-win64-gpl-shared-vulkan\bin\ffmpeg.exe"
    writer = animation.writers['ffmpeg']
    writer = writer(fps=60, metadata=dict(artist='Slimane MZERGUAT'), bitrate=1800)
    ani = FuncAnimation(fig, update, blit=True, frames=int(len(U)/50),interval=200,repeat=False)#int(len(U)/50)
    ani.save("res.mp4", writer=writer)
    plt.show()
if False:# fluctuation analysis
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

if False:#studying the parameters at the boundary at stability 
    U,V,Pr = pickle.load(open("U_V_field-std_0.01.p","rb"))
    u = U[-1]
    v = V[-1]
    p = Pr[-1]
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(6,12))

    ax1r = ax1.twinx()
    ax1r.plot(np.arange(0,N)/N,p[0],label='p',color='g');ax1.plot(np.nan,label='p',color='g')
    ax1 .plot(np.arange(0,N)/N,u[0],label='u',color='r')    
    ax1 .plot(np.arange(0,N)/N,v[0],label='v',color='b')
    ax1 .set_title('at the left wall')
    ax1 .legend()
    ax1 .grid()
    ax1r.tick_params(axis='y', colors='g')
    
    
    ax2r =ax2.twinx()
    ax2r.plot(np.arange(0,N)/N,p[-1],label='p',color='g');ax2.plot(np.nan,label='p',color='g')
    ax2 .plot(np.arange(0,N)/N,u[-1],label='u',color='r')    
    ax2 .plot(np.arange(0,N)/N,v[-1],label='v',color='b')
    ax2 .set_title('at the right wall')
    ax2 .legend()
    ax2 .grid()
    ax2r.tick_params(axis='y', colors='g')
    
    ax3r =ax3.twinx()
    ax3r.xaxis.label.set_color('g')
    ax3r.plot(np.arange(0,N)/N,p[:,int(len(u)/2)],label='p',color='g');ax3.plot(np.nan,label='p',color='g')
    ax3 .plot(np.arange(0,N)/N,u[:,int(len(u)/2)],label='u',color='r')    
    ax3 .plot(np.arange(0,N)/N,v[:,int(len(u)/2)],label='v',color='b')
    ax3 .set_title('at the y=0.5mm')
    ax3 .legend(loc=(0.8,0.5))
    ax3 .grid()
    ax3r.tick_params(axis='y', colors='g')
    plt.show()

if False:#measuring eneries
    U,V,Pr = pickle.load(open("U_V_field-std_0.01.p","rb"))
    u = U[-1]
    v = V[-1]
    p = Pr[-1]
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(6,12))

    ax1r = ax1.twinx()
    ax1r.plot(np.arange(0,N)/N,p[0]*dx*dy,label='p',color='g');ax1.plot(np.nan,label='Pressure energy=$P\\cdot dV$',color='g')
    ax1 .plot(np.arange(0,N)/N,1/2*rho*(u[0]**2+v[0]**2),label='Kenitic energy =$\\frac{1}{2}\\rho \\vec u^2$',color='r')    
    ax1 .set_title('at the left wall')
    ax1 .legend(loc=(0.8,0.5))
    ax1 .grid()
    ax1r.tick_params(axis='y', colors='g')
    ax1 .tick_params(axis='y', colors='r')
    
    
    ax2r =ax2.twinx()
    ax2r.plot(np.arange(0,N)/N,p[-1]*dx*dy,label='p',color='g');ax2.plot(np.nan,label='Pressure energy=$P\\cdot dV$',color='g')
    ax2 .plot(np.arange(0,N)/N,1/2*rho*(u[-1]**2+v[-1]**2),label='Kenitic energy =$\\frac{1}{2}\\rho \\vec u^2$',color='r')
    ax2 .set_title('at the right wall')
    ax2 .legend()
    ax2 .grid()
    ax2r.tick_params(axis='y', colors='g')
    ax2 .tick_params(axis='y', colors='r')
    
    ax3r =ax3.twinx()
    ax3r.xaxis.label.set_color('g')
    ax3r.plot(np.arange(0,N)/N,p[:,int(len(u)/2)]*dx*dy,label='p',color='g');ax3.plot(np.nan,label='Pressure energy=$P\\cdot dV$',color='g')
    ax3 .plot(np.arange(0,N)/N,1/2*rho*(u[:,int(len(u)/2)]**2+v[:,int(len(u)/2)]**2),label='Kenitic energy =$\\frac{1}{2}\\rho \\vec u^2$',color='r') 
    ax3 .set_title('at the y=0.5mm')
    ax3 .legend(loc=(0.8,0.5))
    ax3 .grid()
    ax3r.tick_params(axis='y', colors='g')
    ax3 .tick_params(axis='y', colors='r')
    plt.show()




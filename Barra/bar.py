import numpy as np
#import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

font={'size': 6}
matplotlib.rc('font',**font)
f0=figure(num=1,figsize=(12,12))

ax01=subplot2grid((1,1),(0,0))
ax01.set_title('Posicion vs tiempo ')
ax01.set_ylim(-2.5,2)
ax01.set_xlim(0,20)
ax01.grid(True)
ax01.set_xlabel("time (s)")
ax01.set_ylabel("poition (m) ")

def dydt(X,t):

    k  = 190000   #[N/m]
    I  = 3443.05  #[Kg*m^2]
    L  = 2.78     #[m]
    m  = 1795     #[kg]
    #l0 = 0.4      #[m]
    g  = 9.8      #[m/s^2]
    y0 = 0.4
    l0 = y0 + (m*g)/k
    
    x     =  X[0]
    dx    =  X[1]
    y     =  X[2]
    dy    =  X[3]
    theta =  X[4]
    dtheta=  X[5] 

    delta_x1 = L/2 + x - L/2 * np.cos(theta)
    delta_x2 = x + L/2 * np.cos(theta) - L/2
    delta_y1 = y + y0 - L/2 * np.sin(theta)
    delta_y2 = y + y0 + L/2 * np.sin(theta)
    delta_l1 = np.sqrt(delta_x1**2+delta_y1**2)
    delta_l2 = np.sqrt(delta_x2**2+delta_y2**2)
    #cos_alfa = delta_x1/delta_l1
    #sen_alfa = delta_y1/delta_l1
    

    d2x=(-k*delta_x1*(1-l0/delta_l1)-k*delta_x2*(1-l0/delta_l2))/m
    d2y=(-k*delta_y1*(1-l0/delta_l1)-k*delta_y2*(1-l0/delta_l2)-m*g)/m
    d2theta=(-k*(L/2*np.sin(theta)*delta_x1-L/2*np.cos(theta)*delta_y1)*(1-l0/delta_l1)-k*(1-l0/delta_l2)*(L/2*np.cos(theta)*delta_y2-L/2*np.sin(theta)*delta_x2))/I

    return [dx,d2x,dy,d2y,dtheta,d2theta]

t0=0.0
tf=20.0
pasos=666.6
t=np.linspace(t0,tf,pasos)
init=[0.5,0.5,1.0,2.0,3*np.pi/180,3*np.pi/180]
y=odeint(dydt,init,t,args=())

p011, = ax01.plot(t,y[:,0],'b-',label="x")
p012, = ax01.plot(t,y[:,2],'g-',label="y")
p013, = ax01.plot(t,y[:,4],'r-',label="theta")

ax01.legend([p011,p012,p013],[p011.get_label(),p012.get_label(),p013.get_label()])


plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

font = {'size' : 9}
matplotlib.rc('font',**font)

f0=figure(num=1,figsize=(15,15))
f0.suptitle("Half-Car model",fontsize=12)

ax01=subplot2grid((2,2),(0,0))
ax02=subplot2grid((2,2),(1,0))
ax03=subplot2grid((2,2),(0,1))
ax04=subplot2grid((2,2),(1,1))


ax01.set_title('Posicion vs tiempo')
ax01.set_ylim(-5,10)
ax01.set_xlim(0,5)
ax01.grid(True)
ax01.set_xlabel("time (s)")
ax01.set_ylabel("posicion (cm)")

ax02.set_title('Velocidad vs Tiempo')
ax02.set_ylim(-5,10)
ax02.set_xlim(0,5)
ax02.grid(True)
ax02.set_xlabel("Time (s)")
ax02.set_ylabel("Velocidad (cm/s)")

ax03.set_title('Posicion vs tiempo')
ax03.set_ylim(-5,10)
ax03.set_xlim(0,5)
ax03.grid(True)
ax03.set_xlabel("time (s)")
ax03.set_ylabel("posicion (cm)")

ax04.set_title('Velocidad vs Tiempo')
ax04.set_ylim(-5,15)
ax04.set_xlim(0,5)
ax04.grid(True)
ax04.set_xlabel("Time (s)")
ax04.set_ylabel("Velocidad (cm/s)")

def yp(X,t,u1,du1,u2,du2):

    m1 = 87.15
    m2 = 140.4
    m3 = 1795
    
    k1 = 190000
    kf = 36350
    kr = 26530
    k2 = 190000
    
    c1 = 0.0
    cf = 1200
    cr = 1100
    c2 = 0.0
    
    J  = 3443.05
    b  = 1.32
    c  = 1.46
    v  = 15.0
    td = (b+c)/v

    z1  = X[0]
    dz1 = X[1]
    z2  = X[2]
    dz2 = X[3]
    z3  = X[4]
    dz3 = X[5]
    theta  = X[6]
    dtheta = X[7]

    d2z1 = (kf*(z3+b*theta-z1)+cf*(dz3+b*dtheta-dz1)-k1*(z1-u1(t+td))-c1*(dz1-du1(t+td)))/m1

    d2z2 = (kr*(z3-c*theta-z2)+cr*(dz3-c*dtheta-dz2)-k2*(z2-u2(t))-c2*(dz2-du2(t)))/m2

    d2z3 = (-kr*(z3-c*theta-z2)-cr*(dz3-c*dtheta-dz2)-kf*(z3+b*theta-z1)-cf*(dz3+b*dtheta-dz1))/m3 
 
    d2theta = (c*kr*(z3-c*theta-z2)+c*cr*(dz3-c*dtheta-dz2)-b*kf*(z3+b*theta-z1)-b*cf*(dz3+b*dtheta-dz1))/J

    return [dz1,d2z1,dz2,d2z2,dz3,d2z3,dtheta,d2theta]

def ypa(X,t,u1,du1,u2,du2,f):

    h=0.4
    
    m1 = 87.15
    m2 = 140.4
    m3 = 1795
    
    k1 = 190000
    kf = 36350
    kr = 26530
    k2 = 190000
    
    c1 = 0.0
    cf = 1200
    cr = 1100
    c2 = 0.0
    
    J  = 3443.05
    b  = 1.32
    c  = 1.46
    v  = 15.0
    td = (b+c)/v

    z1  = X[0]
    dz1 = X[1]
    z2  = X[2]
    dz2 = X[3]
    z3  = X[4]
    dz3 = X[5]
    theta  = X[6]
    dtheta = X[7]

    d2z1 = (kf*(z3+b*theta-z1)+cf*(dz3+b*dtheta-dz1)-k1*(z1-u1(t+td))-c1*(dz1-du1(t+td))-h/(b+c)*f(t))/m1

    d2z2 = (kr*(z3-c*theta-z2)+cr*(dz3-c*dtheta-dz2)-k2*(z2-u2(t))-c2*(dz2-du2(t))+h/(b+c)*f(t))/m2

    d2z3 = (-kr*(z3-c*theta-z2)-cr*(dz3-c*dtheta-dz2)-kf*(z3+b*theta-z1)-cf*(dz3+b*dtheta-dz1))/m3 
 
    d2theta = (c*kr*(z3-c*theta-z2)+c*cr*(dz3-c*dtheta-dz2)-b*kf*(z3+b*theta-z1)-b*cf*(dz3+b*dtheta-dz1))/J

    return [dz1,d2z1,dz2,d2z2,dz3,d2z3,dtheta,d2theta]

#Diferentes Caminos o excitaciones

def y1(t):
    v = 15.0
    lamda = 1000
    omega = (2*np.pi*v)/lamda
    A = 0.5
    return A*np.sin(omega*t)

def dy1(t):
    v = 15.0
    lamda = 1000
    omega = (2*np.pi*v)/lamda
    A = 0.5
    return A*omega*np.cos(omega*t)

def y2(t):
    v = 15.0
    lamda = 1000
    omega = (2*np.pi*v)/lamda
    A = 0.5
    return A*omega*np.cos(omega*t)

def dy2(t):
    v = 15.0
    lamda = 1000
    omega = (2*np.pi*v)/lamda
    A = 0.5
    return A*omega*np.cos(omega*t)

def a1(t):
    return 0.4*t

def da1(t):
    return 0.4

def a2(t):
    return 0.4*t

def da2(t):
    return 0.4


def b1(t):
    if t<10:
        return 0.3*t
    else:
        return 0.3*10

def db1(t):
    if t<10:
        return 0.3
    else:
        return 0.0

def b2(t):
    if t<10:
        return 0.3*t
    else:
        return 0.3*10

def db2(t):
    if t<10:
        return 0.3
    else:
        return 0.0    
    
#Fueza motriz como funcion del tiempo "investigar"
    
def force(t):
    if t<10:
        return 10000
    else:
        return 0.0

def force1(t):
    if t<10:
        return 0.0
    else:
        return 100

def force0(t):
    return 0.0


def force2(t):
    return t

t=np.arange(0,8,0.01)

init=[0.01,0.03,0.02,0.04,0.03,0.03,(np.pi*2)/180,(np.pi*2)/180]

y  = odeint(yp,init,t,args=(y1,dy1,y2,dy2))

ya = odeint(ypa,init,t,args=(y1,dy1,y2,dy2,force2))


p011, = ax01.plot(t,y[:,0],'b-',label="z1")
p012, = ax01.plot(t,y[:,2],'g-',label="z2")
p013, = ax01.plot(t,y[:,4],'r-',label="z3")
p014, = ax01.plot(t,y[:,6],'y-',label="theta")

p021, = ax02.plot(t,y[:,1],'b-',label="dz1")
p022, = ax02.plot(t,y[:,3],'g-',label="dz2")
p023, = ax02.plot(t,y[:,5],'r-',label="dz3")
p024, = ax02.plot(t,y[:,7],'y-',label="dtheta")

p031, = ax03.plot(t,ya[:,0],'b-',label="z1")
p032, = ax03.plot(t,ya[:,2],'g-',label="z2")
p033, = ax03.plot(t,ya[:,4],'r-',label="z3")
p034, = ax03.plot(t,ya[:,6],'y-',label="theta")

p041, = ax04.plot(t,ya[:,1],'b-',label="dz1")
p042, = ax04.plot(t,ya[:,3],'g-',label="dz2")
p043, = ax04.plot(t,ya[:,5],'r-',label="dz3")
p044, = ax04.plot(t,ya[:,7],'y-',label="dtheta")

ax01.legend([p011,p012,p013,p014],[p011.get_label(),p012.get_label(),p013.get_label(),p014.get_label()])

ax02.legend([p021,p022,p023,p024],[p021.get_label(),p022.get_label(),p023.get_label(),p024.get_label()])

ax03.legend([p031,p032,p033,p034],[p031.get_label(),p032.get_label(),p033.get_label(),p034.get_label()])

ax04.legend([p041,p042,p043,p044],[p041.get_label(),p042.get_label(),p043.get_label(),p044.get_label()])

plt.show()








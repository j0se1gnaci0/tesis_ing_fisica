import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

font={'size' : 8}
matplotlib.rc('font',**font)

f0=figure(num=1,figsize=(12,12))
f0.suptitle("Quarter-Car Model",fontsize=12)

ax01=subplot2grid((2,1),(0,0))
ax02=subplot2grid((2,1),(1,0))
#ax03=subplot2grid((3,1),(2,0))

#Subtitulos en cada grafico

ax01.set_title('Posicion vs tiempo')
ax01.set_ylim(-5,10)
ax01.set_xlim(0,10)
ax01.grid(True)
ax01.set_xlabel("time (s)")
ax01.set_ylabel("posicion (cm)")

ax02.set_title('Velocidad vs tiempo')
ax02.set_ylim(-10,10)
ax02.set_xlim(0,15)
ax02.grid(True)
ax02.set_xlabel("Time (s)")
ax02.set_ylabel("Velocidad (cm/s)")

#ax03.set_title('Excitacion vs tiempo')
#ax03.set_ylim(-20,20)
#ax03.set_xlim(0,60)
#ax03.grid(True)
#ax03.set_xlabel("time (s)")
#ax03.set_ylabel("Altura (cm)")

def derivada(X,t,u1,du1,u2,du2):

    m1 = 87.15      #[Kg]
    m2 = 140.4      #[Kg]
    m3 = 1795       #[Kg]
    k1 = 190000     #[N/m]
    kf = 36350      #[N/m] 
    kr = 26530      #[N/m]
    k2 = 190000     #[N/m] 
    c1 = 0.0        #[Ns/m]
    cf = 1200       #[Ns/m]
    cr = 1100       #[Ns/m]
    c2 = 0.0        #[Ns/m]
    J  = 3443.05    #[Kg*m^2]
    b  = 1.32       #[m]
    c  = 1.46       #[m]
    v  = 15.0       #[m/s]
    td = (b+c)/v    #[s]

    z1  = X[0]
    dz1 = X[1]
    z2  = X[2]
    dz2 = X[3]
    z3  = X[4]
    dz3 = X[5]
    theta = X[6]
    dtheta= X[7]
    d2z1=(kf*(z3+b*theta-z1)+cf*(dz3+b*dtheta-dz1)-k1*(z1-u1(t+td))-c1*(dz1-du1(t+td)))/m1
    d2z2=(kr*(z3-c*theta-z2)+cr*(dz3-c*dtheta-dz2)-k2*(z2-u2(t))-c2*(dz2-du2(t)))/m2
    d2z3=(-kr*(z3-c*theta-z2)-cr*(dz3-c*dtheta-dz2)-kf*(z3+b*theta-z1)-cf*(dz3+b*dtheta-dz1))/m3
    d2theta=(c*kr*(z3-c*theta-z2)+c*cr*(dz3-c*dtheta-dz2)-b*kf*(z3+b*theta-z1)-b*cf*(dz3+b*dtheta-dz1))/J

    return [dz1,d2z1,dz2,d2z2,dz3,d2z3,dtheta,d2theta]

# Camino sinusoidal

def y1(t):
    w=0.5
    a0=0.5
    return a0*np.sin(w*t)

def dy1(t):
    w=0.5
    a0=0.5
    return a0*w*np.cos(w*t)

def y2(t):
    w=0.5
    a0=0.5
    return a0*np.sin(w*t)

def dy2(t):
    w=0.5
    a0=0.5
    return a0*w*np.cos(w*t)

# Camino con pendiente 0.3
def a1(t):
    return 0.3*t

def da1(t):
    return 0.3

def a2(t):
    return 0.3*t

def da2(t):
    return 0.3

#Camino con pendiente 0.1 luego recto 

def c1(t):
    if t<10:
        return 0.5*t
    else:
        return 10*0.5

def dc1(t):
    if t<10:
        return 0.1
    else:
        return 0.0
    
def c2(t):
    if t<10:
       return 0.5*t
    else:
        return 0.5*10

def dc2(t):
    if t<20:
        return 0.5
    else:
        return 0.0

#Camino plano sin pendiente 

def d1(t):
    return 0.0

def dd1(t):
    return 0.0

def d2(t):
    return 0.0

def dd2(t):
    return 0.0
    
t=np.arange(0,20,0.01)
init=[0.01,0.03,0.02,0.04,0.03,0.03,(np.pi*2)/180,(np.pi*2)/180]

#y=odeint(derivada,init,t,args=(b,db)) (!!!ojo!!!)
y=odeint(derivada,init,t,args=(c1,dc1,c2,dc2))


p011, =ax01.plot(t,y[:,0],'b-',label="z1")
p012, =ax01.plot(t,y[:,2],'g-', label="z2")
p013, =ax01.plot(t,y[:,4],'r',label="z3")
p014, =ax01.plot(t,y[:,6],'y',label="theta")

p021, =ax02.plot(t,y[:,1],'b-',label="dz1")
p022, =ax02.plot(t,y[:,3],'g-',label="dz2")
p023, =ax02.plot(t,y[:,5],'r-',label="dz3")
p024, =ax02.plot(t,y[:,7],'y',label="dtheta")

#p031, =ax03.plot(t,d1(t),'b-',label="excitacion_delantera")
#p032, =ax03.plot(t,d2(t),'g-',label="excitacion_trasera")

ax01.legend([p011,p012,p013,p014],[p011.get_label(),p012.get_label(),p013.get_label(),p014.get_label()])
ax02.legend([p021,p022,p023,p024],[p021.get_label(),p022.get_label(),p023.get_label(),p024.get_label()])
#ax03.legend([p031,p032],[p031.get_label(),p032.get_label()])

#plt.plot(t,y[:,7],'b',t,y[:,2],'r',t,y[:,4],'g',t,y[:,6],'y')

plt.plot(t,y[:,7])
plt.show()




    

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
ax02=subplot2grid((2,2),(0,1))
ax03=subplot2grid((2,2),(1,0))
ax04=subplot2grid((2,2),(1,1))


ax01.set_title('Posicion vs tiempo')
ax01.set_ylim(-5,10)
ax01.set_xlim(0,1)
ax01.grid(True)
ax01.set_xlabel("time (s)")
ax01.set_ylabel("posicion (cm)")

ax02.set_title('Posicion vs Tiempo')
ax02.set_ylim(-5,10)
ax02.set_xlim(0,0.5)
ax02.grid(True)
ax02.set_xlabel("Time (s)")
ax02.set_ylabel("posicion (cm)")

ax03.set_title('Velocidad vs tiempo')
ax03.set_ylim(-5,10)
ax03.set_xlim(0,2)
ax03.grid(True)
ax03.set_xlabel("time (s)")
ax03.set_ylabel("Velocidad (cm/s)")

ax04.set_title('Velocidad vs Tiempo')
ax04.set_ylim(-5,15)
ax04.set_xlim(0,2)
ax04.grid(True)
ax04.set_xlabel("Time (s)")
ax04.set_ylabel("Velocidad (cm/s)")


def dy(X,t,zfl,dzfl,zfr,dzfr,zrl,dzrl,zrr,dzrr):

    ksf = 35000
    csf = 1000
    kusf= 190000
    cusf= 10

    ksr = 38000
    csr = 1000
    kusr= 190000
    cusr= 10

    muf = 59
    mur = 59
    ms  = 1500

    Ixx = 460
    Iyy = 2160

    a = 1.4
    b = 1.7
    s = 3.1
    
    #Ruedas delanteras :rueda izquierda
    zufl = X[0]
    dzufl= X[1]
    #Rueda derecha
    zufr = X[2]
    dzufr= X[3]

    #Ruedas Traseras
    zurl = X[4]
    dzurl= X[5]

    zurr = X[6]
    dzurr= X[7]

    #Chassis
    zs = X[8]
    dzs= X[9]

    phi = X[10]
    dphi= X[11]

    theta = X[12]
    dtheta= X[13]

    #Rueda frontal izquierda

    d2zufl = (ksf*(zs+s/2*phi-a*theta-zufl)+csf*(dzs+s/2*dphi-a*dtheta-dzufl)-kusf*(zufl-zfl(t))-cusf*(dzufl-dzfl(t)))/muf

    #Rueda frontal derecha

    d2zufr = (ksf*(zs-s/2*phi-a*theta-zufr)+csf*(dzs-s/2*dphi-a*dtheta-dzufr)-kusf*(zufr-zfr(t))-cusf*(dzufr-dzfr(t)))/muf
    
    #Rueda Trasera izquierda

    d2zurl = (ksr*(zs+s/2*phi+b*theta-zurl)+csr*(dzs+s/2*dphi+b*dtheta-dzurl)-kusr*(zurl-zrl(t))-cusr*(dzurl-dzrl(t)))/mur

    #Rueda Trasera Derecha

    d2zurr = (ksr*(zs-s/2*phi+b*theta-zurr)+csr*(dzs-s/2*dphi+b*dtheta-dzurr)-kusr*(zurr-zrr(t))-cusr*(dzurr-dzrr(t)))/mur

    #Chassis

    d2zs = (-ksf*(zs+s/2*phi-a*theta-zufl)-csf*(dzs+s/2*dphi-a*dtheta-dzufl)-ksf*(zs-s/2*phi-a*theta-zufr)-csf*(dzs-s/2*dphi-a*dtheta-dzufr)-ksr*(zs+s/2*phi+b*theta-zurl)-csr*(dzs+s/2*dphi+b*dtheta-dzurl))/ms

    #Torques en Phi

    d2phi = (s/2*ksf*(zs-s/2*phi-a*theta-zufr)+s/2*csf*(dzs-s/2*dphi-a*dtheta-dzufr)+s/2*ksr*(zs-s/2*phi+b*theta-zurr)+s/2*csr*(dzs-s/2*dphi+b*dtheta-dzurr)-s/2*ksf*(zs+s/2*phi-a*theta-zufl)-s/2*csf*(dzs+s/2*dphi-a*dtheta-dzufl)-s/2*ksr*(zs+s/2*phi+b*theta-zurl)-s/2*csr*(dzs+s/2*dphi+b*dtheta-dzurl))/Ixx     

    d2theta = (a*ksf*(zs+s/2*phi-a*theta-zufl)+a*csf*(dzs+s/2*dphi-a*dtheta-dzufl)+a*ksf*(zs-s/2*phi-a*theta-zufr)+a*csf*(dzs-s/2*dphi-a*dtheta-dzufr)-b*ksr*(zs+s/2*phi+b*theta-zurl)-b*csr*(dzs+s/2*dphi+b*dtheta-dzurl)-b*ksr*(zs-s/2*phi+b*theta-zurr)-b*csr*(dzs-s/2*dphi+b*dtheta-dzurr))/Iyy

    return [dzufl,d2zufl,dzufr,d2zufr,dzurl,d2zurl,dzurr,d2zurr,dzs,d2zs,dphi,d2phi,dtheta,d2theta]

#Primer camino 1

def zfl1(t):
    return 0.2*t

def dzfl1(t):
    return 0.2

def zfr1(t):
    return 0.4*t

def dzfr1(t):
    return 0.4

def zrl1(t):
    return 0.2*t

def dzrl1(t):
    return 0.2

def zrr1(t):
    return 0.4*t

def dzrr1(t):
    return 0.4

#Segundo Camino 2

def zfl1(t):
    return 0.0

def dzfl1(t):
    return 0.0

def zfr1(t):
    return 0.0

def dzfr1(t):
    return 0.0

def zrl1(t):
    return 0.0

def dzrl1(t):
    return 0.0

def zrr1(t):
    return 0.0

def dzrr1(t):
    return 0.0

#Tercer Camino 3

def zfl1(t):
    if t<5:
        return t
    else:
        return 5.0

def dzfl1(t):
    if t<5:
        return 1.0
    else:
        return 0.0

def zfr1(t):
    if t<5:
        return t
    else:
        return 5.0

def dzfr1(t):
    if t<5:
        return 1.0
    else:
        return 0.0

def zrl1(t):
    if t<5:
        return t
    else:
        return 5.0

def dzrl1(t):
    if t<5:
        return 1.0
    else:
        return 0.0

def zrr1(t):
    if t<5:
        return t
    else:
        return 5.0

def dzrr1(t):
    if t<5:
        return 1.0
    else:
        return 0.0

#Cuarto Camino

#def zfl1(t):
 #   if t<5:
 #       return t
 #   else:
#        return 5.0

#def dzfl1(t):
#    if t<5:
#        return 1.0
#    else:
 #       return 0.0

#def zfr1(t):
#    if t<5:
 #       return 0.3*t
#    else:
#        return 0.3*5

#def dzfr1(t):
#    if t<5:
#        return 0.3
#    else:
#        return 0.0

#def zrl1(t):
#    if t<5:
#        return t
#    else:
#        return 5.0

#def dzrl1(t):
#    if t<5:
#        return 1.0
#    else:
#        return 0.0

#def zrr1(t):
#    if t<5:
#        return 0.3*t
 #   else:
#        return 0.3*5

#def dzrr1(t):
#    if t<5:
#        return 0.3
#    else:
#        return 0.0

#Quinto camino

#def zfl1(t):
#    w0=100
#    return np.sin(w0*t)

#def dzfl1(t):
#    w0=100
#    return np.cos(w0*t)*w0

#def zfr1(t):
#    w0=150
#    return np.cos(w0*t)
    
#def dzfr1(t):
#    w0=150
#    return -w0*np.sin(w0*t)
    
#def zrl1(t):
#    w0=100
#    return np.sin(w0*t)
    
#def dzrl1(t):
#    w0=100
#    return w0*np.cos(w0*t)
    
#def zrr1(t):
#    w0=150
#    return np.cos(w0*t)
    
#def dzrr1(t):
#    w0=150
#    return -w0*np.sin(w0*t)


t = np.arange(0,10,0.0001)
init = [0.01,0.02,0.02,0.01,0.01,0.01,0.02,0.01,0.01,0.02,(np.pi*2)/180,0.8,(np.pi*3)/180,0.5]

y = odeint(dy,init,t,args=(zfl1,dzfl1,zfr1,dzfr1,zrl1,dzrl1,zrr1,dzrr1))

#plt.plot(t,y[:,0],'b-',t,y[:,2],'g-',t,y[:,4],'r-',t,y[:,6],t,y[:,8],'y-',t,y[:,10],t,y[:,12])
#plt.plot(t,y[:,1])

p011, = ax01.plot(t,y[:,0],'b-',label="zfl")
p012, = ax01.plot(t,y[:,2],'g-',label="zfr")
p013, = ax01.plot(t,y[:,4],'r-',label="zrl")
p014, = ax01.plot(t,y[:,6],'y-',label="zrr")
p015, = ax01.plot(t,y[:,8],'k-',label="zs")

p021, = ax02.plot(t,y[:,10],'b-',label="phi")
p022, = ax02.plot(t,y[:,12],'r-',label="theta")

p031, = ax03.plot(t,y[:,1],'b-',label="dzfl")
p032, = ax03.plot(t,y[:,3],'g-',label="dzfr")
p033, = ax03.plot(t,y[:,5],'r-',label="dzrl")
p034, = ax03.plot(t,y[:,7],'y-',label="dzrr")
p035, = ax03.plot(t,y[:,9],'k',label="dzrr")


p041, = ax04.plot(t,y[:,11],'b-',label="dphi")
p042, = ax04.plot(t,y[:,13],'r-',label="dtheta")


ax01.legend([p011,p012,p013,p014,p015],[p011.get_label(),p012.get_label(),p013.get_label(),p014.get_label(),p015.get_label()])

ax02.legend([p021,p022],[p021.get_label(),p022.get_label()])

ax03.legend([p031,p032,p033,p034,p035],[p031.get_label(),p032.get_label(),p033.get_label(),p034.get_label(),p035.get_label()])

ax04.legend([p041,p042],[p041.get_label(),p042.get_label()])

plt.show()


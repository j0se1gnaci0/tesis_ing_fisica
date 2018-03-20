import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal

#Simulacion del Vehiculo
#Dinamica en el eje x del vehiculo simulado

def dxdt(X,t,f):
    m1  = 87.15    #87.15 
    m2  = 140.4
    m3  = 1795
    M   = m1+m2+m3
    x   = X[0]
    xp  = X[1]
    xpp = f(t)/M
    return [xp,xpp]


def f1(t):
    return 0.0

#step: 1.0

def f2(t):
    return 1.0

t     = np.arange(0,20,0.01)
init  = [0.0,15.0] #Vector de condicione inicial
xt    = odeint(dxdt,init,t,args=(f1,))

x  = xt[:,0]
dx = xt[:,1]

m1 = 87.15    #87.15 
m2 = 140.4
m3 = 1795
M  = m1+m2+m3
 
d2x = np.zeros(len(t))/M

num0  = [1,0,0,0]     # Ceros de la funcion de transferencia
den0  = [1,10,20,8]   # Polos de la funcion de transferencia
HPF_f = signal.TransferFunction(num0,den0) #Filtro Pasa Alto HPF (Funcion de Transferencia) 

#Entrada fx sale altas frecuencias de fx, sale la primera y segunda integracion y
t1,fx_hf,y = signal.lsim(HPF_f,d2x,t)

xpp = y[:,0]  #Velocidad de la plataforma en x
xp  = y[:,1]  #Posicion de la plataforma en x

plt.figure(1)
plt.plot(t1,fx_hf)  #Graficar las altas frecuencias de la fuerza especifica

plt.figure(2)
plt.plot(t1,xp)

plt.figure(3)
plt.plot(t1,x)

plt.figure(4)
plt.plot(t1,dx)

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
        return 0.5
    else:
        return 0.0
    
def c2(t):
    if t<10:
        return 0.5*t
    else:
        return 0.5*10

def dc2(t):
    if t<10:
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
    
t   = np.arange(0,20,0.01)
init= [0.01,0.03,0.02,0.04,0.03,0.03,(np.pi*2)/180,(np.pi*2)/180]

#y=odeint(derivada,init,t,args=(b,db)) (!!!ojo!!!)
y = odeint(derivada,init,t,args=(c1,dc1,c2,dc2))

#plt.figure(5)
#plt.plot(t,y[:,7])

#d2z3=(-kr*(z3-c*theta-z2)-cr*(dz3-c*dtheta-dz2)-kf*(z3+b*theta-z1)-cf*(dz3+b*dtheta-dz1))/m3
m3 = 1795
b  = 1.32
c  = 1.46
kr = 26530
cr = 1100
kf = 36350
cf = 1200

d2z = (-kr*(y[:,4]-c*y[:,6]-y[:,2])-cr*(y[:,5]-c*y[:,7]-y[:,3])-kf*(y[:,4]+b*y[:,6]-y[:,0])-cf*(y[:,5]+b*y[:,7]-y[:,1]))/m3

t2,az_hf,z_p = signal.lsim(HPF_f,d2z,t)

dzp = z_p[:,0] #Velocidad en z de la plataforma
zp  = z_p[:,1] #Posicion en z de la plataforma 

plt.figure(5)
plt.plot(t2,az_hf) #Plotear la aceleracion filtrada

plt.figure(6)
plt.plot(t2,zp)   #Plotear la posicion en z de la plataforma

num1 = [1,0]   
den1 = [1,1]

HPF_theta = signal.TransferFunction(num1,den1)#Funcion de transferencia

thetap = y[:,7]
t3,thetapunto,theta=signal.lsim(HPF_theta,thetap,t)

plt.figure(7)
plt.plot(t3,thetapunto)

plt.figure(8)
plt.plot(t3,theta)

plt.show()






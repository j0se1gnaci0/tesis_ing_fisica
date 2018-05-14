import numpy as np
#import matplotlib.pypot as plt
from matplotlib.pylab import *


f0=figure(num=0,figsize=(12,8))
f0.suptitle("Simulation Quarter-Car Model",fontsize=12)
ax=subplot2grid((1,1),(0,0))
ax.set_ylim(-1,3)
ax.set_xlim(-0.5,0.5)


#def plotsuspension(x):
z0=0.001             #posicion del camino
z1=0.01             #posicion rueda medida desde la posicin de equilibrio 
z2=0.02             #posicion chassis medida desde su posicion de equilibrio
    
h1 = 0.75 #0.35;          # resting position of unsprung cm
h2 = 1.8;           # resting position of sprung cm
h3 = 0.2;           # height of unsprung mass block
h4 = 0.35;          # height of sprung mass block
w1 = 0.4;           # width of unsprung mass block
w2 = 0.5;           # width of sprung mass block
w3 = 0.1;           # width of tire spring
w4 = 0.15;          # width of suspension spring
w5 = 0.25;          # spring/damper spacing


x0_t=h1+z1-h3/2    #guarda la posicion en z de la base del bloque no suspedido

x0t = np.array([[0],[x0_t]])         #Posicion de la base del bloque no suspendio
x1t = x0t + np.array([[-w1/2],[0]])  #Vertice inferior izquierdo del bloque no suspendido
x2t = x0t + np.array([[-w1/2],[h3]]) #Vertice superior izquierdo del bloque no suspendido
x3t = x0t + np.array([[w1/2],[h3]])  #Vertice superior derecho del bloque no suspendido
x4t = x0t + np.array([[w1/2],[0]])    #Vertice inferior derecho del bloque no suspendido


#plotear bloque de la rueda
bloque1_x=[]
bloque1_y=[]
bloque1_x.append(x0t[0,0])
bloque1_x.append(x1t[0,0])
bloque1_x.append(x2t[0,0])
bloque1_x.append(x3t[0,0])
bloque1_x.append(x4t[0,0])
bloque1_x.append(x0t[0,0])

bloque1_y.append(x0t[1,0])
bloque1_y.append(x1t[1,0])
bloque1_y.append(x2t[1,0])
bloque1_y.append(x3t[1,0])
bloque1_y.append(x4t[1,0])
bloque1_y.append(x0t[1,0])
plt.fill(bloque1_x,bloque1_y,'b')
#plt.show()


#plotear bloque del chassis
x0_b=h2+z2-h4/2                #posicion de la base del bloque suspendido


x0b = np.array([[0],[x0_b]])         #Posicion de la base del bloque no suspendio
x1b = x0b + np.array([[-w2/2],[0]])  #Vertice inferior izquierdo del bloque no suspendido
x2b= x0b + np.array([[-w2/2],[h4]]) #Vertice superior izquierdo del bloque no suspendido
x3b = x0b + np.array([[w2/2],[h4]])  #Vertice superior derecho del bloque no suspendido
x4b = x0b + np.array([[w2/2],[0]])    #Vertice inferior derecho del bloque no suspendido

bloque2_x=[]
bloque2_y=[]
bloque2_x.append(x0b[0,0])
bloque2_x.append(x1b[0,0])
bloque2_x.append(x2b[0,0])
bloque2_x.append(x3b[0,0])
bloque2_x.append(x4b[0,0])
bloque2_x.append(x0b[0,0])

bloque2_y.append(x0b[1,0])
bloque2_y.append(x1b[1,0])
bloque2_y.append(x2b[1,0])
bloque2_y.append(x3b[1,0])
bloque2_y.append(x4b[1,0])
bloque2_y.append(x0b[1,0])

#plt.plot(bloque1_x,bloque1_y,'b',bloque2_x,bloque2_y,'b')

plt.fill(bloque2_x,bloque2_y,'b')
#plotear spring tire

#cuad=np.array([[10,-10,-10,10,10],[10,10,-10,-10,10]])
#cuad=np.array([[10,-10,-15,-10,10,15,10],[10,10,0,-10,-10,0,10]])
#plt.fill(cuad[0,:],cuad[1,:],'b')
plt.plot(0,z0,'ko')
x0_r=z0
#x0_s=h1+z1+h3/2
L1=x0_t-x0_r
x0r=np.array([[0],[x0_r]])
u=L1/9
x1r=x0r+np.array([[0],[u]])
x2r=x0r+np.array([[-w3/2],[3/2*u]])
x3r=x2r+np.array([[w3],[u]])
x4r=x3r+np.array([[-w3],[u]])
x5r=x4r+np.array([[w3],[u]])
x6r=x5r+np.array([[-w3],[u]])
x7r=x6r+np.array([[w3],[u]])
x8r=x7r+np.array([[-w3],[u]])
x9r=x8r+np.array([[w3/2],[u/2]])
x10r=x9r+np.array([[0],[u]])

spring_x=[]
spring_y=[]
spring_x.append(x0r[0,0])
spring_x.append(x1r[0,0])
spring_x.append(x2r[0,0])
spring_x.append(x3r[0,0])
spring_x.append(x4r[0,0])
spring_x.append(x5r[0,0])
spring_x.append(x6r[0,0])
spring_x.append(x7r[0,0])
spring_x.append(x8r[0,0])
spring_x.append(x9r[0,0])
spring_x.append(x10r[0,0])


spring_y.append(x0r[1,0])
spring_y.append(x1r[1,0])
spring_y.append(x2r[1,0])
spring_y.append(x3r[1,0])
spring_y.append(x4r[1,0])
spring_y.append(x5r[1,0])
spring_y.append(x6r[1,0])
spring_y.append(x7r[1,0])
spring_y.append(x8r[1,0])
spring_y.append(x9r[1,0])
spring_y.append(x10r[1,0])

plt.plot(spring_x,spring_y,'k')


#plotear resorte suspendido

x0_s=h1+z1+h3/2
L2=x0_b-x0_s
x0s=np.array([[-w5/2],[x0_s]])
uu=L2/9
x1s=x0s+np.array([[0],[uu]])
x2s=x0s+np.array([[-w4/2],[3/2*uu]])
x3s=x2s+np.array([[w4],[uu]])
x4s=x3s+np.array([[-w4],[uu]])
x5s=x4s+np.array([[w4],[uu]])
x6s=x5s+np.array([[-w4],[uu]])
x7s=x6s+np.array([[w4],[uu]])
x8s=x7s+np.array([[-w4],[uu]])
x9s=x8s+np.array([[w4/2],[uu/2]])
x10s=x9s+np.array([[0],[6.5*u]])

springun_x=[]
springun_y=[]

springun_x.append(x0s[0,0])
springun_x.append(x1s[0,0])
springun_x.append(x2s[0,0])
springun_x.append(x3s[0,0])
springun_x.append(x4s[0,0])
springun_x.append(x5s[0,0])
springun_x.append(x6s[0,0])
springun_x.append(x7s[0,0])
springun_x.append(x8s[0,0])
springun_x.append(x9s[0,0])
springun_x.append(x10s[0,0])

springun_y.append(x0s[1,0])
springun_y.append(x1s[1,0])
springun_y.append(x2s[1,0])
springun_y.append(x3s[1,0])
springun_y.append(x4s[1,0])
springun_y.append(x5s[1,0])
springun_y.append(x6s[1,0])
springun_y.append(x7s[1,0])
springun_y.append(x8s[1,0])
springun_y.append(x9s[1,0])
springun_y.append(x10s[1,0])

plt.plot(springun_x,springun_y,'k')

#plotear la suspension damper
x0d=np.array([[w5/2],[x0_s]])
a=0.7*(h2-h1-h3/2-h4/2)
b=L2-a
c=0.3*w4

x1d=x0d+np.array([[-c],[a]])
x2d=x0d+np.array([[-c],[0]])
x3d=x0d+np.array([[c],[0]])
x4d=x0d+np.array([[c],[a]])

x5d=x0d+np.array([[-c],[b]])
x6d=x0d+np.array([[c],[b]])

x7d=x0d+np.array([[0],[L2]])
x8d=x0d+np.array([[0],[b]])


damper_ux=[]
damper_uy=[]

damper_ux.append(x1d[0,0])
damper_ux.append(x2d[0,0])
damper_ux.append(x3d[0,0])
damper_ux.append(x4d[0,0])
damper_uy.append(x1d[1,0])
damper_uy.append(x2d[1,0])
damper_uy.append(x3d[1,0])
damper_uy.append(x4d[1,0])

plt.plot(damper_ux,damper_uy,'k')

damper_horx=[]
damper_hory=[]

damper_horx.append(x5d[0,0])
damper_horx.append(x6d[0,0])
damper_hory.append(x5d[1,0])
damper_hory.append(x6d[1,0])

plt.plot(damper_horx,damper_hory,'k')

damper_vertx=[]
damper_verty=[]

damper_vertx.append(x7d[0,0])
damper_vertx.append(x8d[0,0])
damper_verty.append(x7d[1,0])
damper_verty.append(x8d[1,0])

plt.plot(damper_vertx,damper_verty,'k')

plt.show()

import numpy as np
#import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation


k  = 1900.   #[N/m]
I  = 3443.05  #[Kg*m^2]
L  = 2.78     #[m]
m  = 1795.     #[kg]
#l0 = 0.4      #[m]
g  = 9.8      #[m/s^2]
y0 = 0.4
l0 = y0 + (m*g)/(2.*k)
alpha =0.

def dydt(X,t):

    x,dx,y,dy,theta,dtheta = X

    delta_x1 = -L/2. + x + L/2. * np.cos(theta)
    delta_x2 = x - L/2. * np.cos(theta) + L/2.
    delta_y1 = y + y0 - L/2. * np.sin(theta)
    delta_y2 = y + y0 + L/2. * np.sin(theta)
    delta_l1 = np.sqrt(delta_x1**2+delta_y1**2)
    delta_l2 = np.sqrt(delta_x2**2+delta_y2**2)

    #cos_alfa = delta_x1/delta_l1
    #sen_alfa = delta_y1/delta_l1
    
    #buenas ecuaciones
    
    #d2x=(k*delta_x1*(1.-l0/delta_l1)+k*delta_x2*(1.-l0/delta_l2))/m - alpha*dx/m
    #d2y=(-k*delta_y1*(1.-l0/delta_l1)-k*delta_y2*(1.-l0/delta_l2)-m*g)/m- alpha*dy/m
    #d2theta=(L*k/2.*(delta_l1-delta_l2))/I

    #Ecuaciones caso perpendicular
    
    d2x = ( k*delta_x1 / delta_l1 * ( delta_l1 + delta_l2 -2*l0 ))/m - alpha*dx/m
    d2y = ( -k*delta_y1 / delta_l1 *( delta_l1 + delta_l2 -2*l0 )- m*g)/m - alpha*dy/m
    d2theta = ( L*k/2.*(delta_l1 - delta_l2))/I          

    return [dx,d2x,dy,d2y,dtheta,d2theta]

t0 = 0.0
tf = 30.0
delta_t = 1./30
t = np.arange(t0,tf,delta_t)
print(len(t))
init = [0,0,0.5,0,0,0]
#init = [0.01,0,0,0,np.pi/40.,0]
sol = odeint(dydt,init,t,args=())

font = {'size': 6}
matplotlib.rc('font',**font)
f0 = figure(num=1,figsize=(12,12))
ax01 = subplot2grid((1,1),(0,0))
ax01.set_ylim(-L,L)
ax01.set_xlim(-L,L)
scale = 0.5
line, = ax01.plot([],[],'o-',lw=2)

def init_anim():
    line.set_data([],[])
    return line,

def animate(i):
    x,dx,y,dy,theta,dtheta = sol[i,:]
    new_data = ([-L/2,x-L/2*np.cos(theta),x+L/2*np.cos(theta),L/2],[-y0,y-L/2*np.sin(theta),y+L/2*np.sin(theta),-y0])
    
    line.set_data(new_data)
    
    return line,

ani = animation.FuncAnimation(f0, animate, len(t), interval=delta_t, blit=True, init_func = init_anim)


#p011, = ax01.plot(t,y[:,0],'b-',label="x")
#p012, = ax01.plot(t,y[:,2],'g-',label="y")
#p013, = ax01.plot(t,y[:,4],'r-',label="theta")

#ax01.legend([p011,p012,p013],[p011.get_label(),p012.get_label(),p013.get_label()])

plt.show()

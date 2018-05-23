import sys, pygame
from pygame.locals import *
from math import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
   
pygame.init()

ventana = pygame.display.set_mode((900,800))

pygame.display.set_caption("Simulacion-auto-2D")

FPS = pygame.time.Clock()

color = pygame.Color(3,8,6)

pygame.draw.line(ventana,color,(0,400),(900,400),8)

auto = pygame.image.load("auto.png")

posx=0
posy=369

#posx=np.zeros()
blanco = pygame.Color(255,255,255)

class Car(object):
      def __init__(self,start_pos=0,v0=1.0,auto='auto.png')
          self.x0=start_pos
          self.y0=369
          self.vi=v0
          
t0=0.0
tf=1.0
pasos = 1/30    #Fijo
x0=0.0          #Condiciones iniciales posicion
v0=0.0          #Condiciones iniciales velocidad 

while True:
        #Blit the track to the background
        #SCREEN.blit(track, (0, 0))
        ventana.fill(blanco)   
        #ventana.blit(auto,(posx[i],posy)) antes
        ventana.blit(auto,(posx,posy))
        #Test if the game has been quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        def dxdt(X,u,t):
    
            Cd = 0.24
    
            rho= 1.225
    
            m  = 700
    
            A  = 5.0
    
            Fp = 30
    
            x   = X[0]
    
            dx  = X[1]
    
            dx2 = (Fp*u-0.5*0*Cd*A*rho*dx*dx)/m
    
            return [dx,dx2]

###antes###
#t=np.arange(0,50,0.01)

#init=[0.0,0.0]
###antes###

        def U(t):
            if t<tf:
               return 0.0
            else:
               return 100000.0

        t=np.arange(t0,tf,pasos)            
        #i=i+1
        #if i == len(posx)-1:
        #  i=0
        x=odeint(dxdt,[x0,v0],t,args=(U,))

        t0+=1
        tf+=1
        x=x[:,0]
        x0=x[-1]
        posx=x0
        
        #v=x[:,1]
        #v0=v[-1]
        print t0
        print x0
        print tf
        FPS.tick(30)
        pygame.display.update()

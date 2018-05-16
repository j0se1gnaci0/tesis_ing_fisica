import sys, pygame
from pygame.locals import *
from math import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
   
pygame.init()

ventana = pygame.display.set_mode((900,800))

pygame.display.set_caption("Simulacion-auto-2D")

color = pygame.Color(3,8,6)

pygame.draw.line(ventana,color,(0,400),(900,400),8)

auto = pygame.image.load("auto.png")

posx,posy=0,369

#posx=np.zeros()
blanco = pygame.Color(255,255,255)

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

t=np.arange(0,50,0.01)
init=[0.0,0.0]

def U(t):
    if t<15:
       return 2000.0
    else:
       return 0.0

x=odeint(dxdt,init,t,args=(U,)) 

posx=x[:,0]
i=0

while True:
        #Blit the track to the background
        #SCREEN.blit(track, (0, 0))
        ventana.fill(blanco)   
        ventana.blit(auto,(posx[i],posy))
        #Test if the game has been quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        i=i+1
        if i == len(posx)-1:
           i=0
            
        pygame.display.update()

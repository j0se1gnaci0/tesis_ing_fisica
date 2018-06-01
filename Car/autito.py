import sys, pygame
from pygame.locals import *
from math import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def ecuacion(X,t,u):

    m=10.0
    x=X[0]
    dx=X[1]
    d2x=u(t)/m

    return [dx,d2x]

def f(t):
    
    return 2.0

def equation(X,t,u):

    Cd = 0.24
    rho= 1.225
    m  = 700
    A  = 5.0
    Fp = 30
    x  = X[0]
    dx = X[1]
    dx2= (Fp*u(t)-0.5*0*Cd*A*rho*dx**2)/m
    
    return [dx,dx2]
    
def integrar(t0,tf,x0,v0,pasos):
    
    t=np.arange(t0,tf,pasos)
    y=odeint(equation,[x0,v0],t,args=(f,))

    x0=y[:,0]
    v0=y[:,1]

    x0=x0[-1]
    v0=v0[-1]

    t0+=0.033
    tf+=0.033
    
    return x0,v0,t0,tf
    
def juego():
    pygame.init()
    timer = pygame.time.Clock()
    ancho = 900
    alto = 900

    ventana = pygame.display.set_mode((alto,ancho))

    pygame.display.set_caption("Vehiculo 2D")

    #FPS = pygame.time.Clock()

    auto = pygame.image.load('car.png')


    t0    = 0.0
    tf    = 33.0/1000
    pasos = 0.01   #fijo
    
    x0 = 0.0     # [pixeles]
    y0 = alto/2  # [pixeles]

    v0 = 10.0 # m/s
    
    blanco = pygame.Color(255,255,255)
    #timer = pygame.time.Clock()
    #timer = pygame.time.Clock()

    while True:
          #timer = pygame.time.Clock()
          ventana.fill(blanco)
          ventana.blit(auto,(x0,y0))
          
          for event in pygame.event.get():
             if event.type == QUIT:
                pygame.quit()
                sys.exit()

             if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                   pygame.quit()
                   sys.exit()

          x0,v0,t0,tf = integrar(t0,tf,x0,v0,pasos)

          print timer.tick() 

          #print v0
          #print x0
          #if v0<=0:
          #   v0=0
          #print timer
          pygame.display.update()
    

juego()

    

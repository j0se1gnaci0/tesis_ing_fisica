import sys, pygame
from pygame.locals import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sistema(X,t):

    #m = 1.0
    g = 10.0
    #kw = 1.0

    y   = X[0]
    dy  = X[1]
    d2y = g     #-kw*dy**2)/m

    return [dy,d2y]

def proceso(t0,tf,y,vy,pasos):

    t = np.linspace(t0,tf,pasos)
    Y = odeint(sistema,[y,vy],t,args=())

    y  = Y[:,0] 
    vy = Y[:,1]
    y  = y[-1]  #[m]
    vy = vy[-1] #[m]
    #y*=4389
    #vy*=4.4
    delta_t=tf-t0

    t0 +=delta_t
    tf +=delta_t
    
    return y,vy,t0,tf
    
    
def caida_libre():
    reloj = pygame.time.Clock()
    FPS = 30
    pygame.init()
    ancho = 1000
    alto = 10000
    
    ventana = pygame.display.set_mode((ancho,alto))

    pygame.display.set_caption("caida libre")

    #pelota = pygame.image.load('ball.jpg')
    pelota = pygame.transform.scale(pygame.image.load('ball.jpg').convert_alpha(), (40,40))
    blanco = pygame.Color(255,255,255)
    
    x = 450
    y = 0

    vy = 0
    
    t0 = 0
    tf = 0.033
    pasos = 10
    
    while True:
          ventana.fill(blanco)
          ventana.blit(pelota,(x,y))

          for event in pygame.event.get():
              if event.type == QUIT:
                 pygame.quit()
                 sys.exit()

              if event.type == KEYDOWN:
                 if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

          y,vy,t0,tf = proceso(t0,tf,y,vy,pasos)
          #y=y*4389
          #vy=vy*4.4
          print (y,vy,tf)
          reloj.tick(FPS)
    
          pygame.display.update()

caida_libre()



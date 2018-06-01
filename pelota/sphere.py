import sys, pygame
from pygame.locals import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sistema(X,t):

    m = 1
    g = 10
    kw = 1

    y   = X[0]
    dy  = X[1]
    d2y = -m*g-kw*dy**2

    return [dy,d2y]

def proceso(t0,tf,y,vy,pasos):

    t = np.arange(t0,tf,pasos)
    Y = odeint(sistema,[y,vy],t,args=())

    y = Y[:,0]
    vy = Y[:,1]
    y = y[-1]
    vy = vy[-1]

    t0 +=tf
    tf +=tf
    
    return y,vy,t0,tf
    
    
def caida_libre():
    reloj = pygame.time.Clock()
    FPS = 30
    pygame.init()
    ancho=900
    alto=900
    
    ventana=pygame.display.set_mode((alto,ancho))

    pygame.display.set_caption("caida libre")

    #pelota = pygame.image.load('ball.jpg')
    pelota=pygame.transform.scale(pygame.image.load('ball.jpg').convert_alpha(), (40,40))
    blanco = pygame.Color(255,255,255)
    
    x = ancho/2
    y = 0

    vy = 1
    
    t0 = 0
    tf = 33.0/1000
    pasos = 0.01
    
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

          y,v,t0,tf = proceso(t0,tf,y,vy,pasos)
          reloj.tick(FPS)
          pygame.display.update()

caida_libre()



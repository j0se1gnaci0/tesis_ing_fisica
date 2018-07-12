import sys, pygame
from pygame.locals import *
from math import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#Vehiculo viaja con Velocidad Constante

def v_cte(X,t):

    m = 100
    x   = X[0]
    dx  = X[1]
    d2x = 0.0/m
    
    return [dx,d2x]

#Vehiculo viaja con Aceleracion Constante

def a_cte(X,t):

    m = 100
    x   = X[0]
    dx  = X[1]
    d2x = 3.0/m 

    return [dx,d2x]
    
#Aceleracion escrita como funcion del tiempo

def a_tiempo(X,t,u):

    m   = 100.0
    x   = X[0]
    dx  = X[1]
    d2x = u(t)/m

    return [dx,d2x]

#Excitacion externa al Sistema

def f(t):
    
    return 3*t

#Sistema de Ecuaciones Diferenciales de Movimiento

def equation(X,t,u):

    Cd  = 0.24
    rho = 1.225
    m   = 700
    A   = 5.0
    Fp  = 30

    x   = X[0]
    dx  = X[1]
    dx2 = (Fp*u(t)-0.5*0*Cd*A*rho*dx**2)/m
    
    return [dx,dx2]

def acelerador(v):

    v += 0.1

    return v 

def freno(v):

    v -= 0.1

    return v

#Funcion que integra las ecuaciones de movimiento

def integrar(t0,tf,x0,v0,pasos):
    
    t = np.linspace(t0,tf,pasos)
    y = odeint(v_cte,[x0,v0],t,args=())  #excitacion args=(f,), sin excitacon args=()

    x0 = y[:,0]
    v0 = y[:,1]

    x0 = x0[-1]
    v0 = v0[-1]

    delta_t = tf-t0

    t0 += delta_t
    tf += delta_t
    
    return x0, v0, t0, tf

def plotear_camino(x1,y1,x2,y2):

    global ancho
    global alto
    global ventana
    global negro

    width = 10

    
    x2 = ancho/2
    y2 = alto/2
    
    pygame.draw.line(superficie,color,(x1,y1),(x2,y2),width)
       
    pygame.draw.line()    


def juego():
    
    pygame.init()
    timer = pygame.time.Clock()    

    ancho = 2000   #ancho pantalla [pixeles]
    alto  = 900    #alto pantalla [pixeles]

    ventana = pygame.display.set_mode((alto,ancho)) 

    pygame.display.set_caption("Vehiculo 2D")   #Titulo en la pantalla

    #FPS = pygame.time.Clock()

    auto = pygame.image.load('porsche.png') #antes con #
    
    #auto = pygame.transform.scale(pygame.image.load('porsche.png').convert_alpha(),(60,60))
    auto = pygame.transform.scale(auto.convert_alpha(),(60,60))

    FPS  = 60.0                                  #Numero de cuadros por segundos
    slow = 100
                  
    t0  = 0.0                                   #Tiempo Inicial
    tf  = FPS/1000/slow                         #Tiempo Final
    pasos = 100                                 #Numeros de pasos de integracion

    dt = tf-t0 / pasos
    
    x0 = 0.0                                    # Posicion inicial en x[m] 
    y0 = alto/2                                 # Posicion inicial en y[m] 

    v0 = 1.0                              # Velocidad inicial del vehiculo [m/s]
    
    blanco = pygame.Color(255,255,255)          #Color Blanco
    negro  = pygame.Color(0,0,0)                 #Color Negro    

    x = 0.0
    y = y0 + 41

    xf = 2000
    yf = y
    
    while True:
          #timer = pygame.time.Clock()

          ventana.fill(blanco)
          
          pygame.draw.line(ventana,negro,(x,y),(xf,yf),20)

          ventana.blit(auto,(x0*4389,y0))
          
          for event in pygame.event.get():

             if event.type == QUIT:
                pygame.quit()
                sys.exit()

             if event.type == pygame.KEYDOWN:

                if event.key == K_ESCAPE:
                   pygame.quit()
                   sys.exit()
                   print "escape"
                   
                elif event.key == K_a:

                     #v0 += dt*v0
                     #v0 +=0.1
                     v0=acelerador(v0)

                     print "a"
                     
                elif event.key == K_s:

                     #v0 -= dt**2*v0
                     #v0 -= 0.1
                     v0=freno(v0)

                     print "s"

                elif event.key == K_z:

                     print "z"

                elif event.key == K_x:

                     print "x"

                elif event.key == K_c:

                     print "c"
                
          x0, v0, t0, tf = integrar(t0,tf,x0,v0,pasos)

          timer.tick(FPS) 

          #print timer
          pygame.display.update()
    
juego()

    

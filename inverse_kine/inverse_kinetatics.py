from mpl_toolkits.mplot3d import axes3d
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from numpy.linalg import inv

#Punta del aspa sistema de coordenadas fijo
Ai = np.array([[-64.5, 64.5,103.0,43.0,-43.0,-103.0 ],
               [-86.6,-86.6, -5.1,98.7, 98.7,  -5.19],
               [  0.0,  0.0,  0.0, 0.0,  0.0,   0.0 ]])

#Vertices V0 sistema de coordenadas Fijo
V0 = np.array([[-57.5,  57.5,  92.5,  35.0,-35.0,-92.5],
               [-73.6, -73.6, -12.9,  86.6, 86.6,-12.9],
               [135.0, 135.0, 135.0, 135.0,135.0,135.0]])

#Posicion de los servo Motores Sistema de Coordenadas Fijos
Oi = np.array([[-44.0, 44.5,93.0,53.0,-53.0,-93.0],
               [-86.6,-86.6,12.1,81.4, 81.4, 12.1],
               [  0.0,  0.0, 0.0, 0.0,  0.0,  0.0]])

Ui = np.array([[ 0.0, 0.0,0.86,0.86,-0.86,-0.86],
               [-1.0,-1.0,0.5 ,0.5 , 0.5 , 0.5],
               [ 0.0, 0.0,0.0 ,0.0 , 0.0 , 0.0]])

#Base superior para graficar 3d
Base_Superior = np.array([[-57.5, 57.5, 92.5, 35.0, -35.0,-92.5,-57.5],
                          [-73.6,-73.6,-12.9, 86.6,  86.6,-12.9,-73.6],
                          [135.0,135.0,135.0, 135.0,135.0,135.0,135.0]])

#Base Inferior para graficar 3d
Base_Inferior = np.array([[-44.0,44.5,93.0,53.0,-53.0,-93.0,-44.0],
                          [-86.6,-86.6,12.1,81.4,81.4, 12.1,-86.6],
                          [  0.0,  0.0, 0.0, 0.0, 0.0,  0.0,  0.0]])

#piernas = np.zeros((3,3))
#Piernas del Robot

primera_pierna = np.array([[-44.0, -64.5, -57.5],
                           [-86.6, -86.6, -73.6],
                           [  0.0,   0.0, 135.0]])

segunda_pierna = np.array([[ 44.5, 64.5, 57.5],
                           [-86.6,-86.6,-73.6],
                           [  0.0,  0.0,135.0]])

tercera_pierna = np.array([[93.0,103.0,92.5],
                           [12.1,-5.1,-12.9],
                           [ 0.0, 0.0,135.0]])

cuarta_pierna = np.array([[53.0,43.0, 35.0],
                          [81.4,98.7, 86.6],
                          [0.0,  0.0,135.0]])

quinta_pierna = np.array([[-53.0,-43.0,-35.0],
                          [ 81.4, 98.7,86.6],
                          [  0.0,  0.0,135.0]])

sexta_pierna = np.array([[-93.0,-103.0,-92.5],
                         [12.1,  -5.19,-12.9],
                         [ 0.0,    0.0,135.0]])

piernas = np.zeros([])

piernas = np.array([[-44.0,-64.5,-57.5, 44.5, 64.5, 57.5,93.0,103.0, 92.5,53.0,43.0, 35.0,-53.0,-43.0,-35.0,-93.0,-103.0 ,-92.5],
                    [-86.6,-86.6,-73.6,-86.6,-86.6,-73.6,12.1, -5.1,-12.9,81.4,98.7, 86.6, 81.4, 98.7, 86.6, 12.1,  -5.19,-12.9],
                    [  0.0,  0.0,135.0,  0.0,  0.0,135.0, 0.0,  0.0,135.0 ,0.0, 0.0,135.0,  0.0,  0.0, 135.0, 0.0,   0.0 ,135.0]])
    
fig = plt.figure(1)


#for i in range(18):

 #   piernas[:,0] = Oi[:,i]
  #  piernas[:,1] = Ai[:,i]
   # piernas[:,2] = V0[:,i]
    #piernas[:,3] = Oi 

#print piernas

plataforma = fig.add_subplot(111,projection='3d')

#plataforma.plot(Base_Superior[0,:],Base_Superior[1,:],Base_Superior[2,:],'b-')

#plataforma.plot(Base_Inferior[0,:],Base_Inferior[1,:],Base_Inferior[2,:],'b-')

#plataforma.plot(primera_pierna[0,:],primera_pierna[1,:],primera_pierna[2,:],'b-')

#plataforma.plot(segunda_pierna[0,:],segunda_pierna[1,:],segunda_pierna[2,:],'b-')

#plataforma.plot(tercera_pierna[0,:],tercera_pierna[1,:],tercera_pierna[2,:],'b-')

#plataforma.plot(cuarta_pierna[0,:],cuarta_pierna[1,:],cuarta_pierna[2,:],'b-')

#plataforma.plot(quinta_pierna[0,:],quinta_pierna[1,:],quinta_pierna[2,:],'b-')

#plataforma.plot(sexta_pierna[0,:],sexta_pierna[1,:],sexta_pierna[2,:],'b-')

plataforma.plot(piernas[0,:],piernas[1,:],piernas[2,:],'b-')
#for i in range(6):
    
   # piernas[:,0] =  O[:,i]
   # piernas[:,1] = A0[:,i]
    #piernas[:,2] = V0[:,i]

    #plataforma.plot(piernas[0,:],piernas[1,:],piernas[2,:],'b')
    #print piernas
    #plt.show()

#Matriz Homogenea

def RT(alpha, beta, gamma, px, py, pz):

    alpha = alpha*np.pi/180.0
    
    beta  = beta*np.pi/180.0
    
    gamma = gamma*np.pi/180.0

    RX = np.matrix([[1 ,            0,             0, 0],
                    [0 ,np.cos(alpha),-np.sin(alpha), 0],
                    [0 ,np.sin(alpha), np.cos(alpha), 0],
                    [0 ,            0,             0, 1]])

    RY = np.matrix([[np.cos(beta),  0, -np.sin(beta), 0],
                    [           0,  1,             0, 0],
                    [np.sin(beta),  0,  np.cos(beta), 0],
                    [           0,  0,             0, 1]])

    RZ = np.matrix([[np.cos(gamma), -np.sin(gamma),  0,  0],
                    [np.sin(gamma),  np.cos(gamma),  0,  0],
                    [            0,              0,  1,  0],
                    [            0,              0,  0,  1]])

    T = np.matrix([[1, 0, 0, px],
                   [0, 1, 0, py],
                   [0, 0, 1, pz],
                   [0, 0, 0,  1]])

    Rt = T*RZ*RY*RX
  
    return Rt

#Vertices Base Movil, con respecto al sistema de referencia Solidario a la plataforma
#Vertices 4*4, multiplica a la matriz Homogenea que es de 4*4.

Vp = np.matrix([[-57.5, 57.5, 92.5, 35.0,-35.0,-92.5,-57.5],
                      [-73.6,-73.6,-12.9, 86.6, 86.6,-12.9,-73.6],
                      [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                      [  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])


Base_Superior = RT(0,0,10,0,0,135)*Vp

Base_Superior = np.array(Base_Superior)

plataforma.plot(Base_Superior[0,:],Base_Superior[1,:],Base_Superior[2,:],'g-')


Vi = Base_Superior[0:3,0:6]

#Manivela, calculo de la distancia del aspa del motor 

ri = Ai-Oi

LMi2 = np.zeros((1,6))

for i in range(6):

    LMi2[:,i] = ri[:,i].dot(ri[:,i])
    
#Biela, calculo de distancia de la barra

li = V0-Ai

LBi2 = np.zeros((1,6))

for i in range(6):

    LBi2[:,i] = li[:,i].dot(li[:,i])


#Angulo servo motor y Manibela
    
cos_phi=np.zeros((1,6))

for i in range(6):

    phi = 90
    phi = phi*np.pi/180.0
    cos_phi[:,i] = np.cos(phi)
    
li = Vi-Ai

Fi = np.zeros((3,1))

Fi = np.matrix(Fi)

Ji = np.zeros((3,3))

Ji = np.matrix(Ji)

Ri=np.zeros((3,1))

Ri=np.matrix(Ri)


concatenar = []

for i in range(6):
    for x in range(4):

        Fi[0,0] = (ri[:,i].dot(ri[:,i])) - LMi2[0][i]
        Fi[1,0] = (li[:,i].dot(li[:,i])) - LBi2[0][i]
        Fi[2,0] = (ri[:,i].dot(Ui[:,i])) - np.linalg.norm(Ui[:,i])*np.linalg.norm(ri[:,i])*cos_phi[0][i]
        
        Ji[0,0] = 2*ri[0][i]  
        Ji[1,0] = 2*li[0][i]
        Ji[2,0] =   Ui[0][i]
 
        Ji[0,1] = 2*ri[1][i]
        Ji[1,1] = 2*li[1][i] 
        Ji[2,1] =   Ui[1][i] 

        Ji[0,2] = 2*ri[2][i]
        Ji[1,2] = 2*li[2][i]
        Ji[2,2] =   Ui[2][i]
        
        Ji_1 = inv(Ji)
    
        Ri = np.matrix([[ri[0][i]],[ri[1][i]],[ri[2][i]]]) - (Ji_1 * Fi)
        
        ri[0][i] = Ri[0,0]
        ri[1][i] = Ri[1,0]
        ri[2][i] = Ri[2,0]

    concatenar.append(Ri)             

#for i in range(len(concatenar)):
    
   # ri.append = np.array(concatenar[i])  


Ai = ri + Oi

#for i in range(6):
    
plt.show()



from mpl_toolkits.mplot3d import axes3d
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import fsolve


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
    

fig = plt.figure(1)

#for i in range(18):

 #   piernas[:,0] = Oi[:,i]
  #  piernas[:,1] = Ai[:,i]
   # piernas[:,2] = V0[:,i]
    #piernas[:,3] = Oi 

#print piernas

plataforma = fig.add_subplot(111,projection='3d')

plataforma.plot(Base_Superior[0,:],Base_Superior[1,:],Base_Superior[2,:],'b-')

plataforma.plot(Base_Inferior[0,:],Base_Inferior[1,:],Base_Inferior[2,:],'b-')

plataforma.plot(primera_pierna[0,:],primera_pierna[1,:],primera_pierna[2,:],'b-')

plataforma.plot(segunda_pierna[0,:],segunda_pierna[1,:],segunda_pierna[2,:],'b-')

plataforma.plot(tercera_pierna[0,:],tercera_pierna[1,:],tercera_pierna[2,:],'b-')

plataforma.plot(cuarta_pierna[0,:],cuarta_pierna[1,:],cuarta_pierna[2,:],'b-')

plataforma.plot(quinta_pierna[0,:],quinta_pierna[1,:],quinta_pierna[2,:],'b-')

plataforma.plot(sexta_pierna[0,:],sexta_pierna[1,:],sexta_pierna[2,:],'b-')


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


Base_Superior = RT(20,0,0,0,0,135)*Vp

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

#Implementacion del Metodo Newton Rhapson:

F = np.zeros((18,1))

#Fi = np.matrix(Fi)

#Ji = np.zeros((3,3))

#Ji = np.matrix(Ji)

#Ri=np.zeros((3,1))

#Ri=np.matrix(Ri)


#concatenar = []

#for i in range(6):
#    for x in range(4):

#        Fi[0,0] = (ri[:,i].dot(ri[:,i])) - LMi2[0][i]
#        Fi[1,0] = (li[:,i].dot(li[:,i])) - LBi2[0][i]
#        Fi[2,0] = (ri[:,i].dot(Ui[:,i])) - np.linalg.norm(Ui[:,i])*np.linalg.norm(ri[:,i])*cos_phi[0][i]
        
#        Ji[0,0] = 2*ri[0][i]  
#        Ji[1,0] = 2*li[0][i]
#        Ji[2,0] =   Ui[0][i]
 
#        Ji[0,1] = 2*ri[1][i]
#        Ji[1,1] = 2*li[1][i] 
#        Ji[2,1] =   Ui[1][i] 

#        Ji[0,2] = 2*ri[2][i]
#        Ji[1,2] = 2*li[2][i]
#        Ji[2,2] =   Ui[2][i]
        
#        Ji_1 = inv(Ji)
    
#        Ri = np.matrix([[ri[0][i]],[ri[1][i]],[ri[2][i]]]) - (Ji_1 * Fi)
        
#        ri[0][i] = Ri[0,0]
#        ri[1][i] = Ri[1,0]
#        ri[2][i] = Ri[2,0]

#    concatenar.append(Ri)             

#for i in range(len(concatenar)):
    
   # ri.append = np.array(concatenar[i])  

#Tenemos Oi fijo Vi movil y encontramos Ai por newton Rhapson

#Ai = ri + Oi

#for i in range(6):

phi = 90.0
phi = (phi*np.pi)/180.0

LM = 20.0

LB = 136.0

def f(z):

    #z=np.array(z)
    
    A1x = z[0] #[0]
    A1y = z[1] #[0]
    A1z = z[2] #[0]

    A2x = z[3] #[1]
    A2y = z[4] #[1]
    A2z = z[5] #[1]
 
    A3x = z[6] #[2]
    A3y = z[7] #[2]
    A3z = z[8] #[2]

    A4x = z[9] #[3]
    A4y = z[10] #[3]
    A4z = z[11] #[3]

    A5x = z[12] #[4]
    A5y = z[13] #[4]
    A5z = z[14] #[4]

    A6x = z[15] #[5]
    A6y = z[16] #[5]
    A6z = z[17] #[5]

    
    O1x = Oi[0][0]
    O1y = Oi[1][0]
    O1z = Oi[2][0]

    O2x = Oi[0][1]
    O2y = Oi[1][1]
    O2z = Oi[2][1]
 
    O3x = Oi[0][2]
    O3y = Oi[1][2]
    O3z = Oi[2][2]

    O4x = Oi[0][3]
    O4y = Oi[1][3]
    O4z = Oi[2][3]

    O5x = Oi[0][4]
    O5y = Oi[1][4]
    O5z = Oi[2][4]

    O6x = Oi[0][5]
    O6y = Oi[1][5]
    O6z = Oi[2][5]

    U1x = Ui[0][0]
    U1y = Ui[1][0]
    U1z = Ui[2][0]

    U2x = Ui[0][1]
    U2y = Ui[1][1]
    U2z = Ui[2][1]
 
    U3x = Ui[0][2]
    U3y = Ui[1][2]
    U3z = Ui[2][2]

    U4x = Ui[0][3]
    U4y = Ui[1][3]
    U4z = Ui[2][3]

    U5x = Ui[0][4]
    U5y = Ui[1][4]
    U5z = Ui[2][4]

    U6x = Ui[0][5]
    U6y = Ui[1][5]
    U6z = Ui[2][5]

    V1x = Vi[0][0]
    V1y = Vi[1][0]
    V1z = Vi[2][0]

    V2x = Vi[0][1]
    V2y = Vi[1][1]
    V2z = Vi[2][1]
 
    V3x = Vi[0][2]
    V3y = Vi[1][2]
    V3z = Vi[2][2]

    V4x = Vi[0][3]
    V4y = Vi[1][3]
    V4z = Vi[2][3]

    V5x = Vi[0][4]
    V5y = Vi[1][4]
    V5z = Vi[2][4]

    V6x = Vi[0][5]
    V6y = Vi[1][5]
    V6z = Vi[2][5]

    F  = np.zeros(18)
    
    F[0]  = (A1x-O1x)**2+(A1y-O1y)**2+(A1z-O1z)**2-LM**2
    F[1]  = (V1x-A1x)**2+(V1y-A1y)**2+(V1z-A1z)**2-LB**2
    F[2]  = (A1x-O1x)*U1x+(A1y-O1y)*U1y+(A1z-O1z)*U1z-LM*np.cos(phi)
    F[3]  = (A2x-O2x)**2+(A2y-O2y)**2+(A2z-O2z)**2-LM**2
    F[4]  = (V2x-A2x)**2+(V2x-A2y)**2+(V2z-A2z)**2-LB**2
    F[5]  = (A2x-O2x)*U2x+(A2y-O2y)*U2y+(A2z-O2z)*U2z-LM*np.cos(phi)
    F[6]  = (A3x-O3x)**2+(A3y-O3y)**2+(A3z-O3z)**2-LM**2
    F[7]  = (V3x-A3x)**2+(V3x-A3y)**2+(V3z-A3z)**2-LB**2
    F[8]  = (A3x-O3x)*U3x+(A3y-O3y)*U3y+(A3z-O3z)*U3z-LM*np.cos(phi)
    F[9]  = (A4x-O4x)**2+(A4y-O4y)**2+(A4z-O4z)**2-LM**2
    F[10] = (V4x-A4x)**2+(V4x-A4y)**2+(V4z-A4z)**2-LB**2
    F[11] = (A4x-O4x)*U4x+(A4y-O4y)*U4y+(A4z-O4z)*U4z-LM*np.cos(phi)
    F[12] = (A5x-O5x)**2+(A5y-O5y)**2+(A5z-O5z)**2-LM**2
    F[13] = (V5x-A5x)**2+(V5x-A5y)**2+(V5z-A5z)**2-LB**2
    F[14] = (A5x-O5x)*U5x+(A5y-O5y)*U5y+(A5z-O5z)*U5z-LM*np.cos(phi)
    F[15] = (A6x-O6x)**2+(A6y-O6y)**2+(A6z-O6z)**2-LM**2
    F[16] = (V6x-A6x)**2+(V6x-A6y)**2+(V6z-A6z)**2-LB**2
    F[17] = (A6x-O6x)*U6x+(A6y-O6y)*U6y+(A6z-O6z)*U6z-LM*np.cos(phi)
    
    return F


Ai=[-64.5,-86.6,0.0,64.5,-86.6,0,103.0,-5.1,0,43,98.7,0,-43,98.7,0,-103,-5.19,0]


A=fsolve(f,Ai)


print A

plt.show()



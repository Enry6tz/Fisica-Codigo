import cv2
import time
import numpy as np
import sympy
import sympy.integrals

import matplotlib.pyplot as plt

def calcularVelocidades (p0,p1,p2,deltaT):
    vx = (p0[0] - p1[0])/deltaT
    vy = -(p0[1] - p1[1])/deltaT

    vx2 = (p1[0] - p2[0])/deltaT
    vy2 = -(p1[1] - p2[1])/deltaT

    return vx,vy,vx2,vy2

def calcularAceleraciones (vx,vy,vx2,vy2,deltaT):
    ax = (vx-vx2)/deltaT
    ay = (vy-vy2)/deltaT

    return ax,ay

# Set up tracker.
tracker = cv2.TrackerCSRT_create()
# Read video
video = cv2.VideoCapture("Basket1.mp4")

# Read first frame.
ok, frame = video.read()

# Define an initial bounding box
bbox = (345, 403, 25, 25)
#bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

print("bound inicial:")
print(bbox[0],bbox[1],bbox[2],bbox[3])
xinicial=bbox[0]+(bbox[2]/2)
yinicial=bbox[1]+(bbox[3]/2)

#Para determinar la velocidad vamos a comparar 2 puntos y para la aceleracion comparamos 2 velocidades
p0=(xinicial,yinicial)
p1=(0,0)
p2=(0,0)

tiempoInicial = time.perf_counter()
tiempoPrev = tiempoInicial
numtick = 0

lAvgAX = []
lAvgAY = []
lAvgVX = []
lAvgVY = []

lT_TOTAL = []
lX_TOTAL = []
lY_TOTAL = []
lX_TOTAL2 = []
lY_TOTAL2 = []

lAvgAY_TOTAL = []
lAvgAX_TOTAL = []
lAvgVX_TOTAL = []

cut = False
while True:
    numtick+=1
    # Read a new frame
    _, frame = video.read()
    
    if (cut):
        break

    # Update tracker
    _, bbox = tracker.update(frame)

    if (10<numtick<47):
        
        centre1 = (int(bbox[0]), int(bbox[1]))
        centre2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, centre1, centre2, (255,0,0), 2, 1)
        p2=p1
        p1=p0
        p0=(int(bbox[0]+(bbox[2]/2)),int(bbox[1]+(bbox[3]/2)))
        lX_TOTAL.append(p0[0])
        lY_TOTAL.append(720-p0[1])
        if(p2[0]!=0): #cuando los 3 puntos existan
            tiempoActual = time.perf_counter()-tiempoInicial
            deltaT = tiempoActual-tiempoPrev
            tiempoPrev = tiempoActual
            lT_TOTAL.append(tiempoActual)
            vx,vy,_,_ = calcularVelocidades(p0,p1,p2,deltaT)
            ax,ay = calcularAceleraciones(*calcularVelocidades(p0,p1,p2,deltaT),deltaT)

            lAvgAX.append(ax)
            lAvgAY.append(ay)
            lAvgVX.append(vx)
            lAvgVY.append(vy)

            lAvgAY_TOTAL.append(ay)
            lAvgAX_TOTAL.append(ax)
            lAvgVX_TOTAL.append(vx)

            #esmitacion promedio (a mejorar)
            vx = sum(lAvgVX)/len(lAvgVX)
            vy = sum(lAvgVY)/len(lAvgVY)
            ax = sum(lAvgAX)/len(lAvgAX)
            ay = sum(lAvgAY)/len(lAvgAY)

            maximo = 6
            if(len(lAvgVX)>maximo):
                lAvgVX.pop(0)
            if(len(lAvgVY)>maximo):
                lAvgVY.pop(0)
            if(len(lAvgAX)>maximo):
                lAvgAX.pop(0)
            if(len(lAvgAY)>maximo):
                lAvgAY.pop(0)
            
            thetaV = np.arctan(vy/(vx+0.01))
            thetaA = np.arctan(ay/(ax+0.01))
            magV = np.sqrt(vx**2+vy**2)
            magA = np.sqrt(ax**2+ay**2)
            
            if(numtick<46):
                #defino las funciones que describen a X respecto a T               
                tx = sympy.Symbol("tx")
                ax_t = ax.__round__(2)
                vx_t = sympy.integrate(ax_t,tx) + vx.__round__(2)
                rx_t = sympy.integrate(vx_t,tx) + xinicial 
                aceleracionX = sympy.lambdify(tx,ax_t)
                velocidadX = sympy.lambdify(tx,vx_t)
                trayectoriaX = sympy.lambdify(tx,rx_t)

                #defino las funciones que describen a Y respecto a T
                ty = sympy.Symbol("ty")
                ay_t = ay.__round__(2)
                vy_t = sympy.integrate(ay_t,ty) + vy.__round__(2)
                ry_t = sympy.integrate(vy_t,ty) + yinicial 
                aceleracionY = sympy.lambdify(ty,ay_t)
                velocidadY = sympy.lambdify(ty,vy_t)
                trayectoriaY = sympy.lambdify(ty,ry_t)
            else:
                # si estoy en el ultimo tick saco el promedio final
                ax = sum(lAvgAX_TOTAL)/len(lAvgAX_TOTAL)
                ay = sum(lAvgAY_TOTAL)/len(lAvgAY_TOTAL)
                vx = sum(lAvgVX_TOTAL)/len(lAvgVX_TOTAL)
             
                tx = sympy.Symbol("tx")
                ax_t = ax.__round__(2)
                vx_t = sympy.integrate(ax_t,tx) + vx.__round__(2)
                rx_t = sympy.integrate(vx_t,tx) + xinicial 
                aceleracionX = sympy.lambdify(tx,ax_t)
                velocidadX = sympy.lambdify(tx,vx_t)
                trayectoriaX = sympy.lambdify(tx,rx_t)

                ty = sympy.Symbol("ty")
                ay_t = ay.__round__(2)
                vy_t = sympy.integrate(ay_t,ty) + vy.__round__(2)
                ry_t = sympy.integrate(vy_t,ty) + yinicial 
                aceleracionY = sympy.lambdify(ty,ay_t)
                velocidadY = sympy.lambdify(ty,vy_t)
                trayectoriaY = sympy.lambdify(ty,ry_t)
                cut = True

            
            #vectores con coordenadas
            cv2.arrowedLine(frame,p0,(int(p0[0]+vx),int(p0[1]+vy)),(160,0,160),3)
            cv2.arrowedLine(frame,p0,(int(p0[0]+ax),int(p0[1]+ay)),(230,230,0),3)

            stringA_T = "a(t): {}"
            stringV_T = "v(t): {}"
            stringR_T = "r(t): {}"
            stringAcc = "a({:0.2f}): {:0.2f}"
            stringVel = "v({:0.2f}): {:0.2f}"
            stringTra = "r({:0.2f}): {:0.2f}"

            print(f"(t: {tiempoActual:0.2f}, x:{p0[0]}, y:{p0[1]})")
            print("Descripcion de X respecto a T:")
            print(f"{stringA_T.format(ax_t):26}", end='')
            print(f"{stringV_T.format(vx_t):26}", end='')
            print(f"{stringR_T.format(rx_t):26}")
            print(f"{stringAcc.format(float(tiempoActual),aceleracionX(float(tiempoActual))):26}", end='')
            print(f"{stringVel.format(float(tiempoActual),velocidadX(float(tiempoActual))):26}", end='')
            print(f"{stringTra.format(float(tiempoActual),trayectoriaX(float(tiempoActual))):26}", end='\n'*2)

            print("Descripcion de Y respecto a T:")
            print(f"{stringA_T.format(ay_t):26}", end='')
            print(f"{stringV_T.format(vy_t):26}", end='')
            print(f"{stringR_T.format(ry_t):26}")
            print(f"{stringAcc.format(float(tiempoActual),aceleracionY(float(tiempoActual))):26}", end='')
            print(f"{stringVel.format(float(tiempoActual),velocidadY(float(tiempoActual))):26}", end='')
            print(f"{stringTra.format(float(tiempoActual),trayectoriaY(float(tiempoActual))):26}", end='\n'*2)

            if(cut):
                cv2.waitKey(-1)
                break
                       
    # else :
    #     # Tracking failure
    #     cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display result
    cv2.imshow("GinobiLib", frame)
    # dominioX = np.linspace(0,25,500) #500 puntos de 0 a 15  
#np.arrange(0,15,0.1) x puntos separados por 0.1 (15/0.1 puntos)

    

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

cv2.imshow("GinobiLib", frame)
prediccionX = []
prediccionY = []
for i in range(len(lT_TOTAL)):
    prediccionX.append(trayectoriaX(lT_TOTAL[i]))
    prediccionY.append(trayectoriaY(lT_TOTAL[i]))

plt.plot(prediccionX,prediccionY)
plt.scatter(lX_TOTAL,lY_TOTAL)
plt.show()  
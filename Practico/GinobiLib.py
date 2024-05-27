import cv2
import time
import numpy as np
import pandas as pd
import sympy

import matplotlib.pyplot as plt

def calcularVelocidades (p0,p1,p2,deltaT):
    vx = (p0[0] - p1[0])/deltaT
    vy = (p0[1] - p1[1])/deltaT

    vx2 = (p1[0] - p2[0])/deltaT
    vy2 = (p1[1] - p2[1])/deltaT

    return vx,vy,vx2,vy2

def calcularAceleraciones (vx,vy,vx2,vy2,deltaT):
    ax = (vx-vx2)/deltaT
    ay = (vy-vy2)/deltaT

    return ax,ay

tickInicio = 0
tickFin = 40
bbox = (0,0,0,0)
print("1) Tiro_encesto1\n2) Tiro_encesto2\n3) Tiro_encesto3\n4) Tiro_erro1")
seleccion = int(input("Seleccione video: "))

match(seleccion):
    case 1:
        tickInicio = 2
        tickFin = 37
        bbox = (304,493,44,48)
        videoSeleccionado = "Tiro_encesto1.mp4"
    case 2:
        tickInicio = 1
        tickFin = 37
        bbox = (483,476,35,36)
        videoSeleccionado = "Tiro_encesto2.mp4"
    case 3:
        tickInicio = 9
        tickFin = 47
        bbox = (345, 403, 25, 25)
        videoSeleccionado = "Tiro_encesto3.mp4"
    case 4:
        tickInicio = 0
        tickFin = 40
        bbox = (496,528,35,39)
        videoSeleccionado = "Tiro_erro1.mp4"

tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture("Practico\\"+videoSeleccionado)
ok, frame = video.read()

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

print("bound inicial:")
print(bbox[0],bbox[1],bbox[2],bbox[3])
xinicial=int(bbox[0]+(bbox[2]/2))
yinicial=int(bbox[1]+(bbox[3]/2))
vxinicial = 0
vyinicial = 0
ayinicial = 0
ancho = video.get(cv2.CAP_PROP_FRAME_WIDTH)
alto = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

#Para determinar la velocidad vamos a comparar 2 puntos y para la aceleracion comparamos 2 velocidades
pInicial =(xinicial,yinicial)
p0 = pInicial
p1=(0,0)
p2=(0,0)

tiempoInicial = time.perf_counter()
tiempoPrev = tiempoInicial
numtick = 0
crearIniciales = False
crearVelocidades = False

lAvgAX = []
lAvgAY = []
lAvgVX = []
lAvgVY = []

lT_TOTAL = []
lX_TOTAL = []
lY_TOTAL = []
lAY_TOTAL = []
lAX_TOTAL = []
lVX_TOTAL = []

lTodosLosResultados = [] # lista de septuplas de la forma(t,x,y,vx,vy,ax,ay)
cut = False
while True:
    numtick+=1
    # Read a new frame
    ok, frame = video.read()
    
    if (cut):
        break

    # Update tracker
    ok, bbox = tracker.update(frame)

    if (ok and tickInicio<numtick<tickFin):
        centre1 = (int(bbox[0]), int(bbox[1]))
        centre2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, centre1, centre2, (255,0,0), 2, 1)
        p2=p1
        p1=p0
        p0=(int(bbox[0]+(bbox[2]/2)),int(bbox[1]+(bbox[3]/2)))
        if(p2[0]!=0): #cuando los 3 puntos 
            if(not crearIniciales):
                xinicial=p0[0]
                yinicial=p0[1]
                crearIniciales = True
            tiempoActual = time.perf_counter()-tiempoInicial
            deltaT = tiempoActual-tiempoPrev
            tiempoPrev = tiempoActual
            lT_TOTAL.append(tiempoActual.__round__(3))
            vx,vy,vx2,vy2 = calcularVelocidades(p0,p1,p2,deltaT)
            ax,ay = calcularAceleraciones(vx,vy,vx2,vy2,deltaT)

            lAvgAX.append(ax.__round__(3))
            lAvgAY.append(ay.__round__(3))
            lAvgVX.append(vx.__round__(3))
            lAvgVY.append(vy.__round__(3))

            lAY_TOTAL.append(ay.__round__(3))
            lAX_TOTAL.append(ax.__round__(3))
            lVX_TOTAL.append(vx.__round__(3))

            lX_TOTAL.append(p0[0])
            lY_TOTAL.append(video.get(4)-p0[1])
            
            #lTodosLosResultados.append((tiempoActual.__round__(2),p0[0],p0[1],vx.__round__(2),vy.__round__(2),ax.__round__(2),ay.__round__(2)))

            # estimacion promedio (a mejorar)
            maximo = 8
            if(len(lAvgVX)>maximo):
                lAvgVX.pop(0)
                lAvgVY.pop(0)
                lAvgAX.pop(0)
                lAvgAY.pop(0)
                vx = sum(lAvgVX)/(len(lAvgVX)+1)
                vy = sum(lAvgVY)/(len(lAvgVY)+1)
                # ax = sum(lAvgAX)/(len(lAvgAX)+1)
                ay = sum(lAvgAY)/(len(lAvgAY)+1)
                if(not crearVelocidades):
                    vxinicial=vx
                    vyinicial=vy
                    crearVelocidades = True

            #vectores con coordenadas
            cv2.arrowedLine(frame,p0,(int(p0[0]+vx/2),int(p0[1]+vy/2)),(160,0,160),3)
            cv2.arrowedLine(frame,p0,(int(p0[0]),int(p0[1]+ay/2)),(230,230,0),3)
            if(numtick>=tickFin-1):
                # si estoy en el ultimo tick saco el promedio final
                ayinicial = sum(lAY_TOTAL[3:-3])/len(lAY_TOTAL[3:-3])

                tx = sympy.Symbol("tx")
                ax_t = 0
                vx_t = vxinicial.__round__(2)
                rx_t = sympy.integrate(vx_t,tx) + xinicial 
                aceleracionX = sympy.lambdify(tx,ax_t)
                velocidadX = sympy.lambdify(tx,vx_t)
                trayectoriaX = sympy.lambdify(tx,rx_t)

                ty = sympy.Symbol("ty")
                ay_t = ayinicial.__round__(2)
                vy_t = sympy.integrate(ay_t,ty) + vyinicial.__round__(2)
                ry_t = sympy.integrate(vy_t,ty) + yinicial 
                aceleracionY = sympy.lambdify(ty,ay_t)
                velocidadY = sympy.lambdify(ty,vy_t)
                trayectoriaY = sympy.lambdify(ty,ry_t)
                stringA_T = "a(t): {}"
                stringV_T = "v(t): {}"
                stringR_T = "r(t): {}"
                stringAcc = "a({:0.2f}): {:0.2f}"
                stringVel = "v({:0.2f}): {:0.2f}"
                stringTra = "r({:0.2f}): {:0.2f}"
                # separar los strings cada 26 caracteres
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
                cut=True

            if(cut):
                break
            
    # Display result


    cv2.imshow("GinobiLib", frame)
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        break

cv2.imshow("GinobiLib", frame)

dominio = np.linspace(0,tiempoActual+1,250)

plt.figure(figsize=(7,5))
plt.subplot(121)
plt.plot(dominio,trayectoriaY(dominio),label="r(t)",color='b')
plt.title("TrayectoriaY")
plt.xlabel("Tiempo/segs")
plt.ylabel("Distancia/px")
plt.xlim((-0.5,tiempoActual))
plt.ylim((alto,0))
plt.subplot(121)

k=0.9
#thetaV = np.arctan((-trayectoriaY(k)-trayectoriaY(k-0.02))/(trayectoriaX(k)-trayectoriaX(k-0.02)))
thetaV = np.arctan(-velocidadY(k)/velocidadX(k))
# thetaV = np.pi/2
plt.quiver(k,trayectoriaY(k), np.cos(thetaV),np.sin(thetaV),scale=1,scale_units='xy',color="r")
plt.quiver(k,trayectoriaY(k), 0,-ay/4,scale=1,scale_units='xy',color="g")

plt.legend()
plt.show()


plt.figure(figsize=(7,5))
plt.subplot(121)
plt.plot(dominio,velocidadY(dominio),label="v(t)",color='r')
plt.axhline(ayinicial,label="a(t)",color='g')
plt.ylim(0,alto)
plt.show()

plt.ylim(0,alto)
plt.xlim(0, ancho)
plt.scatter(trayectoriaX(dominio),alto-trayectoriaY(dominio))
plt.scatter(lX_TOTAL,lY_TOTAL)
plt.show()

# Crear un DataFrame con los datos
data = {
    'Tiempo': lT_TOTAL,
    'PosX': lX_TOTAL,
    'PosY': lY_TOTAL,
    'VelX': lVX_TOTAL,
    'VelY': lAY_TOTAL,
    'AcelY': lAY_TOTAL
}
# Imprimir las longitudes de cada lista
for key, value in data.items():
    print(f"Length of {key}: {len(value)}")

# Alinear las longitudes de las listas
max_length = max(len(lst) for lst in data.values())

for key in data:
    if len(data[key]) < max_length:
        data[key].extend([None] * (max_length - len(data[key])))

df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo CSV
stringcsv = f"datos_{videoSeleccionado}.csv"
df.to_csv(stringcsv, index=False)

# for i in range(2,len(lX_TOTAL)):
#   p2=(lX_TOTAL[i-2],lY_TOTAL[i-2])
#   p1=(lX_TOTAL[i-1],lY_TOTAL[i-1])
#   p0=(lX_TOTAL[i],lY_TOTAL[i])
#   deltaT = lT_TOTAL[i]-lT_TOTAL[i-1]
#   vx,vy,vx2,vy2 = calcularVelocidades(p0,p1,p2,deltaT)
#   ax,ay = calcularAceleraciones(vx,vy,vx2,vy2,deltaT)

#   lVX_TOTAL.append(vx.__round__(2))
#   lVY_TOTAL.append(vy.__round__(2))
#   lAX_TOTAL.append(ax.__round__(2))
#   lAY_TOTAL.append(ay.__round__(2))

# # deberiamos aplicar la teoria del error aca
# # ax = sum(lAvgAX_TOTAL)/len(lAvgAX_TOTAL)

# ayinicial = np.mean(lAY_TOTAL)
# vxinicial = np.mean(lVX_TOTAL)
# vyinicial=0
# sum=0;
# for i in range(1,5):
#   sum+=1
#   vyinicial+=lVY_TOTAL[i]
# vyinicial/=sum

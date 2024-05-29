import cv2
import time
import numpy as np
import pandas as pd
import sympy
import matplotlib.pyplot as plt

from Tracker import Tracker


''' 
    Definimos las funciones que vamos a usar para obtener la
    velocidad entre 2 puntos y la aceleracion entre 2 velocidades
'''
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
tickFin = 0
bbox = (0,0,0,0)
print("Seleccione que video desea analizar:")
print("1) Tiro_encesto1")
print("2) Tiro_encesto2")
print("3) Tiro_encesto3")
print("4) Tiro_erro1")
seleccion = int(input("Seleccion: "))

match(seleccion):
    case 1:
        videoSeleccionado = "Tiro_encesto1.mp4"
    case 2:
        videoSeleccionado = "Tiro_encesto2.mp4"
    case 3:
        videoSeleccionado = "Tiro_encesto3.mp4"
    case 4:
        videoSeleccionado = "Tiro_erro1.mp4"

mostrar = input("Desea ver el video en ejecucion? (Y/N): ")
show = True
if(str(mostrar) == 'Y' or str(mostrar) == 'y'):
    show = True
else:
    show = False
deltaT = 1/30
tracker = Tracker()
tracker.track(videoSeleccionado,deltaT,show)

'''
    En estas listas se guardan todos los datos
    que luego se exportan en un archivo '.csv'
'''
lX_TOTAL,lY_TOTAL,lPos_TOTAL,lT_TOTAL = tracker.getLists()
ALTURA,ANCHO = tracker.getSize()
lY_INVERTIDA = []
for i in range(len(lY_TOTAL)):
    lY_INVERTIDA.append(ALTURA-lY_TOTAL[i])
vxinicial = 0
vyinicial = 0
ayinicial = 0
alturaMaxima = 0
tiempo_alturaMaxima = 0

lAY_TOTAL = []
lAX_TOTAL = []
lVX_TOTAL = []
lVY_TOTAL = []
for i in range(2,len(lPos_TOTAL)):

    vx,vy,vx2,vy2 = calcularVelocidades(lPos_TOTAL[i-2],lPos_TOTAL[i-1],lPos_TOTAL[i],deltaT)
    ax,ay = calcularAceleraciones(vx,vy,vx2,vy2,deltaT)
    
    lVX_TOTAL.append(vx.__round__(3))
    lVY_TOTAL.append(vy.__round__(3))
    lAX_TOTAL.append(ax.__round__(3))
    lAY_TOTAL.append(ay.__round__(3))

for i in range(len(lY_TOTAL)):
    if(alturaMaxima<lY_TOTAL[i]):
        alturaMaxima = lY_TOTAL[i]
        tiempo_alturaMaxima = lT_TOTAL[i]

xinicial = lX_TOTAL[0]
yinicial = lY_TOTAL[0]
ayinicial = sum(lAY_TOTAL[10:-10])/len(lAY_TOTAL[10:-10])
vyinicial = -ayinicial * tiempo_alturaMaxima
##vyinicial = sum(lVY_TOTAL[:6])/len(lVY_TOTAL[:6])
vxinicial = (lX_TOTAL[-1]-xinicial)/lT_TOTAL[-1]
moduloV = np.sqrt(vxinicial**2 + vyinicial**2)
anguloTiro = np.arccos(vxinicial/moduloV)
tiempoFinal = lT_TOTAL[-1]

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
print(f"(t: {tiempoFinal:0.2f}, x:{lX_TOTAL[-1]}, y:{lY_TOTAL[-1]})")
print("Descripcion de X respecto a T:")
print(f"{stringA_T.format(ax_t):26}", end='')
print(f"{stringV_T.format(vx_t):26}", end='')
print(f"{stringR_T.format(rx_t):26}")
print(f"{stringAcc.format(float(tiempoFinal),aceleracionX(float(tiempoFinal))):26}", end='')
print(f"{stringVel.format(float(tiempoFinal),velocidadX(float(tiempoFinal))):26}", end='')
print(f"{stringTra.format(float(tiempoFinal),trayectoriaX(float(tiempoFinal))):26}", end='\n'*2)

print("Descripcion de Y respecto a T:")
print(f"{stringA_T.format(ay_t):26}", end='')
print(f"{stringV_T.format(vy_t):26}", end='')
print(f"{stringR_T.format(ry_t):26}")
print(f"{stringAcc.format(float(tiempoFinal),aceleracionY(float(tiempoFinal))):26}", end='')
print(f"{stringVel.format(float(tiempoFinal),velocidadY(float(tiempoFinal))):26}", end='')
print(f"{stringTra.format(float(tiempoFinal),trayectoriaY(float(tiempoFinal))):26}", end='\n'*2)
print(f"Angulo inicial de tiro: {float(anguloTiro):0.2f}")

dominio = np.linspace(0,tiempoFinal+1,250)

plt.figure(figsize=(7,5))
plt.subplot(121)
plt.plot(dominio,trayectoriaY(dominio),label="r(t)",color='b')
plt.title("TrayectoriaY")
plt.xlabel("Tiempo/segs")
plt.ylabel("Distancia/px")
plt.xlim((-0.5,tiempoFinal))
plt.ylim((ALTURA,0))
plt.subplot(121)

#thetaV = np.arctan((-trayectoriaY(k)-trayectoriaY(k-0.02))/(trayectoriaX(k)-trayectoriaX(k-0.02)))
for i in range(0,3):
    k=tiempoFinal*((i+1)/6)
    thetaV = np.arctan(-velocidadY(k)/velocidadX(k))
    # thetaV = np.pi/2
    plt.quiver(k,trayectoriaY(k), np.cos(thetaV),np.sin(thetaV),scale=1,scale_units='xy',color="r")
    plt.quiver(k,trayectoriaY(k), 0,ay/10,scale=1,scale_units='xy',color="g")

plt.legend()
plt.show()

plt.figure(figsize=(7,5))
plt.subplot(121)
plt.plot(dominio,velocidadY(dominio),label="v(t)",color='r')
plt.axhline(ayinicial,label="a(t)",color='g')
plt.ylim(0,ALTURA)
plt.show()

plt.ylim(0,ALTURA)
plt.xlim(0, ANCHO)
plt.scatter(trayectoriaX(dominio),ALTURA-trayectoriaY(dominio))
plt.scatter(lX_TOTAL,lY_INVERTIDA)
plt.show()

# Crear un DataFrame con los datos
data = {
    'Tiempo': lT_TOTAL,
    'PosX': lX_TOTAL,
    'PosY': lY_TOTAL,
    'VelX': lVX_TOTAL,
    'VelY': lVY_TOTAL,
    'AcelY': lAY_TOTAL
}
# Imprimir las longitudes de cada lista
# for key, value in data.items():
    # print(f"Length of {key}: {len(value)}")

# Alinear las longitudes de las listas
max_length = max(len(lst) for lst in data.values())

for key in data:
    if len(data[key]) < max_length:
        data[key].extend([None] * (max_length - len(data[key])))

df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo CSV
df.to_csv("Practico\\DataFrames\\df_"+videoSeleccionado[:-4]+".csv", index=False)

import cv2
import time
import numpy as np
import pandas as pd
import sympy
import matplotlib.pyplot as plt


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
        tickInicio = 8
        tickFin = 47
        bbox = (345, 403, 25, 25)
        videoSeleccionado = "Tiro_encesto3.mp4"
    case 4:
        tickInicio = 1
        tickFin = 40
        bbox = (496,528,35,39)
        videoSeleccionado = "Tiro_erro1.mp4"

'''
    Inicializamos el tracker segun el video seleccionado
    acomodando manualmente la posicion inicial del Bounding Box
    del tracker y la cantidad de frames que nos interesan estudiar.
'''

tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture("Practico\\"+videoSeleccionado)
ok, frame = video.read()

ok = tracker.init(frame, bbox)

print(f"Posicion BBOX inicial: [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
xinicial=int(bbox[0]+(bbox[2]/2))
yinicial=int(bbox[1]+(bbox[3]/2))
vxinicial = 0
vyinicial = 0
ayinicial = 0

# Dimensiones del video (resolucion)
ALTURA, ANCHO, _ = frame.shape


#obtenemos un deltaT fijo que depende de la longitud del video y de su framerate
fps = video.get(cv2.CAP_PROP_FPS)
#frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = tickFin-tickInicio
duration = frame_count/fps
deltaT = duration/fps


pInicial =(xinicial,yinicial)
p0 = pInicial
p1=p2=(0,0)

tiempoActual = 0
numtick = 0

alturaMaxima = 0
tiempo_alturaMaxima = 0

'''
    En estas listas se guardan todos los datos
    que luego se exportan en un archivo '.csv'
'''
lAvgAX = []
lAvgAY = []
lAvgVX = []
lAvgVY = []

lT_TOTAL = []
lPos_TOTAL = []
lX_TOTAL = []
lY_TOTAL = []
lAY_TOTAL = []
lAX_TOTAL = []
lVX_TOTAL = []
lVY_TOTAL = []

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

        # Graficamos el bounding box alrededor de la pelota

        centre1 = (int(bbox[0]), int(bbox[1]))
        centre2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, centre1, centre2, (255,0,0), 2, 1)
        p2=p1
        p1=p0
        p0=(int(bbox[0]+(bbox[2]/2)),int(bbox[1]+(bbox[3]/2)))
        
        tiempoActual += deltaT
        lT_TOTAL.append(tiempoActual.__round__(3))

        lPos_TOTAL.append(p0)
        lX_TOTAL.append(p0[0])
        '''
            Como el origen de coordenadas esta en la esquina superior
            izquierda la lista de posiciones en Y esta invertida para
            representar los graficos correctamente
        '''
        lY_TOTAL.append(ALTURA-p0[1])
        
        # Cuando los 3 puntos existan podemos graficar los vectores velocidad y aceleracion
        if(p2[0]!=0):
            vx,vy,vx2,vy2 = calcularVelocidades(p0,p1,p2,deltaT)
            ax,ay = calcularAceleraciones(vx,vy,vx2,vy2,deltaT)

            lAvgAX.append(ax.__round__(3))
            lAvgAY.append(ay.__round__(3))
            lAvgVX.append(vx.__round__(3))
            lAvgVY.append(vy.__round__(3))

            # estimacion usando un promedio entre 6 frames 
            maximo = 6
            if(len(lAvgVX)>maximo):
                lAvgVX.pop(0)
                lAvgVY.pop(0)
                lAvgAX.pop(0)
                lAvgAY.pop(0)
                vx = sum(lAvgVX)/(len(lAvgVX))
                vy = sum(lAvgVY)/(len(lAvgVY))
                # ax = sum(lAvgAX)/(len(lAvgAX)+1)
                ay = sum(lAvgAY)/(len(lAvgAY))

            if(alturaMaxima<p0[1]):
                alturaMaxima = p0[1]
                tiempo_alturaMaxima = tiempoActual

            # Graficamos los vectores
            cv2.arrowedLine(frame,p0,(int(p0[0]+vx/2),int(p0[1]+vy/2)),(160,0,160),3)
            cv2.arrowedLine(frame,p0,(int(p0[0]),int(p0[1]+ay/2)),(230,230,0),3)
        if(numtick>=tickFin-1):
            break

    # Display result
    cv2.imshow("GinobiLib", frame)
    cv2.setWindowProperty("GinobiLib", cv2.WND_PROP_TOPMOST, 1)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        break

cv2.imshow("GinobiLib", frame)

for i in range(2,len(lPos_TOTAL)):

    vx,vy,vx2,vy2 = calcularVelocidades(lPos_TOTAL[i-2],lPos_TOTAL[i-1],lPos_TOTAL[i],deltaT)
    ax,ay = calcularAceleraciones(vx,vy,vx2,vy2,deltaT)
    
    lVX_TOTAL.append(vx.__round__(3))
    lVY_TOTAL.append(vy.__round__(3))
    lAX_TOTAL.append(ax.__round__(3))
    lAY_TOTAL.append(ay.__round__(3))


ayinicial = sum(lAY_TOTAL[10:-10])/len(lAY_TOTAL[10:-10])
vyinicial = -ayinicial * tiempo_alturaMaxima
##vyinicial = sum(lVY_TOTAL[:6])/len(lVY_TOTAL[:6])
vxinicial = (lX_TOTAL[-1]-xinicial)/tiempoActual
moduloV = np.sqrt(vxinicial**2 + vyinicial**2)
anguloTiro = np.arccos(vxinicial/moduloV)

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
print(f"Angulo inicial de tiro: {float(anguloTiro):0.2f}")

dominio = np.linspace(0,tiempoActual+1,250)

plt.figure(figsize=(7,5))
plt.subplot(121)
plt.plot(dominio,trayectoriaY(dominio),label="r(t)",color='b')
plt.title("TrayectoriaY")
plt.xlabel("Tiempo/segs")
plt.ylabel("Distancia/px")
plt.xlim((-0.5,tiempoActual))
plt.ylim((ALTURA,0))
plt.subplot(121)

#thetaV = np.arctan((-trayectoriaY(k)-trayectoriaY(k-0.02))/(trayectoriaX(k)-trayectoriaX(k-0.02)))
for i in range(0,3):
    k=tiempoActual*((i+2)/6)
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
plt.ylim(0,ALTURA)
plt.show()

plt.ylim(0,ALTURA)
plt.xlim(0, ANCHO)
plt.scatter(trayectoriaX(dominio),ALTURA-trayectoriaY(dominio))
plt.scatter(lX_TOTAL,lY_TOTAL)
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

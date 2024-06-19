import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import csv
import tkinter as tk
from tkinter import simpledialog

# Función para seleccionar el video
def seleccionar_video():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    video = simpledialog.askstring("Seleccionar video", "Ingrese el número del video a predecir:\n1) Entra1\n2) Error1\n3) Error2\n4) Error3\n5) Error4")
    return video

# Seleccionar el video
video_seleccionado = seleccionar_video()

# Verificar si el usuario presionó "Cancelar"
if video_seleccionado is None:
    print("Selección de video cancelada.")
    exit()

# Mapear la selección del usuario al nombre del archivo de video
videos = {
    "1": 'Files/Videos/entra1.mp4',
    "2": 'Files/Videos/error1.mp4',
    "3": 'Files/Videos/error2.mp4',
    "4": 'Files/Videos/error3.mp4',
    "5": 'Files/Videos/error4.mp4'
}

# Cargo el video seleccionado
cap = cv2.VideoCapture(videos.get(video_seleccionado, 'Files/Videos/entra1.mp4')) #Alguna opcion invalida corre 'Files/Videos/entra1.mp4' por default

# Crear el objeto Finder con el color de la pelota
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 146, 'smin': 100, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 247}

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1500)]
prediction = False                          # La prediccion arranca en falso
start = True                                # Corre el video
timeStep = 0.033333                         # Intervalo de tiempo entre frames
gravedad = 9.8                              # Aceleración debido a la gravedad (m/s²)

# TODO ESTO PARA CAMBIAR LA TIPOGRAFIA Y FONDO EL ENCESTA O NO
def draw_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, thickness=3, color=(0, 255, 0), shadow_color=(0, 0, 0)):
    x, y = position

    # Draw shadow
    cv2.putText(img, text, (x + 2, y + 2), font, font_scale, shadow_color, thickness + 2, lineType=cv2.LINE_AA)

    # Draw main text
    cv2.putText(img, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

while True:

    if start:
        if len(posListX) == 10:
            start = False              # Este if hace: si ha alcanzado 10 elementos(puntos), para el video
        success, img = cap.read()      # Haciendo falso start y leyendo un frame del video

    # Encuentra el color de la pelota
    imgColor, mask = myColorFinder.update(img, hsvVals)
    # Encuentra la ubicación de la pelota
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    #Posicion del Aro en la pantalla
    pxAroI = (1240, 310)
    pxAroF = (1290, 310)
    # cv2.circle(imgContours, pxAroI, 10, (255, 0, 0), cv2.FILLED)
    # cv2.circle(imgContours, pxAroF, 10, (255, 0, 0), cv2.FILLED)

    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # Regresion Polinomica y = Ax^2 + Bx + C
        # Encuentra los coeficientes

        A, B, C = np.polyfit(posListX, posListY, 2)  # Calculo de los coeficientes con la regresion lineal

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, pos, 10, (255, 0, 0), cv2.FILLED)  # Puntos
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 0, 255), 5)  # Linea que sigue los puntos

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 255, 255), cv2.FILLED)  # Parabola

        if len(posListX) < 10:
            # Predicción
            a = A
            b = B
            c = C - 200

            discriminant = b ** 2 - 4 * a * c

            if discriminant >= 0:
                x1 = int((-b + math.sqrt(discriminant)) / (2 * a))
                # x2 = int((-b - math.sqrt(discriminant)) / (2 * a))

                if 1240 <= x1 <= 1290:
                    prediction = True
                else:
                    prediction = False
            else:
                prediction = False

        if prediction:
            draw_text(imgContours, "Encesta", (50, 150), font=cv2.FONT_HERSHEY_COMPLEX, font_scale=2, thickness=5, color=(0, 200, 0), shadow_color=(0, 0, 0))
        else:
            draw_text(imgContours, "No Encesta", (50, 150), font=cv2.FONT_HERSHEY_COMPLEX, font_scale=2, thickness=5, color=(0, 0, 200), shadow_color=(0, 0, 0))

        # Calcular y dibujar vectores de velocidad y aceleración
        if len(posListX) > 5:
            # Media móvil para la velocidad en X e Y
            vX = np.mean([(posListX[-i] - posListX[-i-1]) / timeStep for i in range(1, 6)])
            vY = np.mean([(posListY[-i] - posListY[-i-1]) / timeStep for i in range(1, 6)])

            # Aceleración en X (debe ser cercana a 0)
            aX = 0

            # Aceleración en Y (debe ser cercana a -9.8 m/s², negativa debido a la gravedad)
            aY = gravedad

            # Dibujar vectores de velocidad en X e Y
            cv2.arrowedLine(imgContours, (posListX[-1], posListY[-1]), (posListX[-1] + int(vX * 0.1), posListY[-1]), (0, 255, 0), 2)
            cv2.arrowedLine(imgContours, (posListX[-1], posListY[-1]), (posListX[-1], posListY[-1] + int(vY * 0.1)), (255, 0, 0), 2)

            # Magnitud de la velocidad
            vMagnitude = math.sqrt(vX ** 2 + vY ** 2)

            if vMagnitude != 0:
                cv2.arrowedLine(imgContours, (posListX[-1], posListY[-1]),
                                (posListX[-1] + int((vX / vMagnitude) * vMagnitude * 0.1),
                                 posListY[-1] + int((vY / vMagnitude) * vMagnitude * 0.1)),
                                (0, 0, 255), 2)

            # Dibujar vectores de aceleración en X e Y
            cv2.arrowedLine(imgContours, (posListX[-1], posListY[-1]), (posListX[-1] + int(aX * 0.1), posListY[-1]), (0, 255, 255), 2)
            cv2.arrowedLine(imgContours, (posListX[-1], posListY[-1]), (posListX[-1], posListY[-1] + int(aY * 0.1)), (255, 255, 0), 2)

            # Magnitud de la aceleración
            aMagnitude = math.sqrt(aX ** 2 + aY ** 2)

            if aMagnitude != 0:
                cv2.arrowedLine(imgContours, (posListX[-1], posListY[-1]),
                                (posListX[-1] + int((aX / aMagnitude) * aMagnitude * 0.8),
                                 posListY[-1] + int((aY / aMagnitude) * aMagnitude * 0.8)),
                                (255, 0, 255), 2)

    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    cv2.imshow("ImageColor", imgContours)

    key = cv2.waitKey(100)
    if key == ord(" "):  # Espacio corre de vuelta el video
        start = True

    # Guardar las coordenadas en un archivo CSV al finalizar el procesamiento
    with open('ball_coordinates.csv', 'w', newline='') as file:
        writer = csv.writer(file)  # Crea el escritor CSV
        writer.writerow(["X", "Y"])  # Escribe los encabezados en la primera fila
        writer.writerows(zip(posListX, posListY))  # Escribe las coordenadas en el archivo CSV

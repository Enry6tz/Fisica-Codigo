import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import csv
import matplotlib.pyplot as plt

# Inicializar el gráfico
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Altura de la Pelota')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Altura (px)')
ax.set_title('Altura de la Pelota en función del Tiempo')
ax.legend()
plt.grid(True)

# Cargo el video seleccionado
cap = cv2.VideoCapture('Files/Videos/Secuencia 02.mp4') # Alguna opcion invalida corre 'entra1.mp4' por default

# Crear el objeto Finder con el color de la pelota
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 91, 'vmin': 121, 'hmax': 179, 'smax': 255, 'vmax': 255}

# Variables
posListX = []            # Inicializa como lista vacía
posListY = []            # Inicializa como lista vacía
time_data = []           # Inicializa como lista vacía
prediction = False       # La prediccion arranca en falso
start = True             # Corre el video
timeStep = 0.016666      # Intervalo de tiempo entre frames
frame_number = 0

while True:
    if start:
        success, img = cap.read()  

        if not success:
            break

        # Encuentra el color de la pelota
        imgColor, mask = myColorFinder.update(img, hsvVals)
        # Encuentra la ubicación de la pelota
        imgContours, contours = cvzone.findContours(img, mask, minArea=500)

        if contours:
            posListX.append(contours[0]['center'][0])
            posListY.append(contours[0]['center'][1])
            time_data.append(frame_number * timeStep)
            frame_number += 1

        if posListX:
            for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                pos = (posX, posY)
                if i == 0:
                    cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
                else:
                    cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (83, 153, 0), 5)  # Linea que sigue los puntos

        # Invertir los valores de Y para que la altura aumente hacia arriba
        max_y = max(posListY)
        y_data_inverted = [max_y - y for y in posListY]

        # Actualizar el gráfico
        line.set_xdata(time_data)
        line.set_ydata(y_data_inverted)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)  # Pausa para permitir la actualización del gráfico

        # Mostrar el video
        imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
        cv2.imshow("ImageColor", imgContours)

    key = cv2.waitKey(1)

    # Salir del loop presionando 'q'
    if key == ord('q'):
        break

# Guardar las coordenadas en un archivo CSV al finalizar el procesamiento
with open('ball_coordinates_energy.csv', 'w', newline='') as file:
    writer = csv.writer(file)  # Crea el escritor CSV
    writer.writerow(["X", "Y"])  # Escribe los encabezados en la primera fila
    writer.writerows(zip(posListX, posListY))  # Escribe las coordenadas en el archivo CSV

cap.release()
cv2.destroyAllWindows()
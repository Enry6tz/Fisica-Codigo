import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
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

# Mapear la selección del usuario al nombre del archivo de video
videos = {
    "1": 'Files/Videos/entra1.mp4',
    "2": 'Files/Videos/error1.mp4',
    "3": 'Files/Videos/error2.mp4',
    "4": 'Files/Videos/error3.mp4',
    "5": 'Files/Videos/error4.mp4'
}

# Obtener el archivo de video seleccionado o usar 'error2.mp4' por defecto si la selección no es válida
video_path = videos.get(video_seleccionado, 'Files/Videos/error2.mp4')

# Cargo el video seleccionado
cap = cv2.VideoCapture(video_path)

# Crear el objeto Finder con el color de la pelota
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 146, 'smin': 100, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 247}

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1500)]
prediction = False                          # La predicción arranca en falso
start = True                                # Corre el video
timeStep = 0.033333                         # Intervalo de tiempo entre frames
gravedad = 9.8                              # Aceleración debido a la gravedad (m/s²)
masa = 0.65                                 # Masa de la pelota en kg

# Constante de conversión de píxeles a metros
pixels_to_metros = 1 / 212  # 1 metro = 219 píxeles

# Listas para almacenar la energía en cada frame
energia_cinetica_lista = []
energia_potencial_lista = []
energia_total_lista = []

# Tamaño de la ventana para la media móvil
window_size = 5

# Función para suavizar datos con media móvil
def media_movil(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Función para ajustar un polinomio y derivarlo
def ajustar_polinomio_y_derivar(x, y, grado):
    p = Polynomial.fit(x, y, grado)
    derivada = p.deriv()
    return derivada

# Función para calcular la energía cinética
def calcular_energia_cinetica(vx, vy):
    return 0.5 * masa * (vx ** 2 + vy ** 2)

# Función para calcular la energía potencial
def calcular_energia_potencial(yPos):
    return masa * gravedad * (yPos - 0.6575342465753424)

# Función para calcular la energía total
def calcular_energia_total(energia_cinetica, energia_potencial):
    return energia_cinetica + energia_potencial

# Función para graficar la energía
def graficar_energia(energia_cinetica_lista, energia_potencial_lista, energia_total_lista):
    plt.plot(range(len(energia_cinetica_lista)), energia_cinetica_lista, marker='o', linestyle='-', color='b', label='Energía Cinética')
    plt.plot(range(len(energia_potencial_lista)), energia_potencial_lista, marker='o', linestyle='-', color='g', label='Energía Potencial')
    plt.plot(range(len(energia_total_lista)), energia_total_lista, marker='o', linestyle='-', color='r', label='Energía Total')
    plt.xlabel('Frame')
    plt.ylabel('Energía')
    plt.title('Evolución de la Energía a través de los Frames')
    plt.grid(True)
    plt.legend()
    plt.show()

# Función para dibujar texto en la imagen
def draw_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, thickness=3, color=(0, 255, 0), shadow_color=(0, 0, 0)):
    x, y = position

    # Draw shadow
    cv2.putText(img, text, (x + 2, y + 2), font, font_scale, shadow_color, thickness + 2, lineType=cv2.LINE_AA)

    # Draw main text
    cv2.putText(img, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

# Bucle principal
while True:
    if start:
        success, img = cap.read()
        if not success:
            break  # Si no se puede leer el siguiente frame, salir del bucle

    # Resto del código para procesar la imagen y detectar la pelota
    imgColor, mask = myColorFinder.update(img, hsvVals)
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)
    
    pxAroI = (1240, 310)
    pxAroF = (1290, 310)
    
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # Polynomial Regression y = Ax^2 + Bx + C
        A, B, C = np.polyfit(posListX, posListY, 2)
        
        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, pos, 10, (255, 0, 0), cv2.FILLED)   # Dibuja puntos
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 0, 255), 5)   # Dibuja líneas

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 255, 255), cv2.FILLED)  # Dibuja la parábola

        if len(posListX) < 10:
            # Predicción
            a = A
            b = B
            c = C - 200

            discriminant = b ** 2 - 4 * a * c

            if discriminant >= 0:
                x1 = int((-b + math.sqrt(discriminant)) / (2 * a))

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
 
    # Suavizar las posiciones antes de calcular la velocidad
    if len(posListX) > window_size:
        posListX_suave = media_movil(posListX, window_size)
        posListY_suave = media_movil(posListY, window_size)

        # Calcular derivadas para obtener velocidades
        if len(posListX_suave) > 5:
            derivadaX = ajustar_polinomio_y_derivar(range(len(posListX_suave)), posListX_suave, 2)
            derivadaY = ajustar_polinomio_y_derivar(range(len(posListY_suave)), posListY_suave, 2)
            
            vX = derivadaX(len(posListX_suave) - 1) * pixels_to_metros / timeStep
            vY = derivadaY(len(posListY_suave) - 1) * pixels_to_metros / timeStep
            
            # Calcular la energía cinética y potencial
            yPos_metros = (posListY_suave[0] - posListY_suave[-1]) * pixels_to_metros  # Inversión y conversión
            energia_cinetica = calcular_energia_cinetica(vX, vY)
            energia_potencial = calcular_energia_potencial(yPos_metros)
            energia_total = calcular_energia_total(energia_cinetica, energia_potencial)
            
            # Agregar a las listas
            energia_cinetica_lista.append(energia_cinetica)
            energia_potencial_lista.append(energia_potencial)
            energia_total_lista.append(energia_total)

    # Mostrar imagen con OpenCV
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    cv2.imshow("ImageColor", imgContours)
    
    key = cv2.waitKey(100)
    
    if key == ord(" "):  # Espacio para avanzar el video
        start = True
    
    if key == ord('q'):  # Salir del bucle si se presiona 'q'
        break

# Mostrar gráfico de energía al finalizar el procesamiento
graficar_energia(energia_cinetica_lista, energia_potencial_lista, energia_total_lista)

# Guardar las coordenadas en un archivo CSV al finalizar el procesamiento
with open('ball_coordinates.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["X", "Y"])
    writer.writerows(zip(posListX, posListY))

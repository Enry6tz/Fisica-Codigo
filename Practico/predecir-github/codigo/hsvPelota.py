import cv2
import numpy as np

# Cargar la nueva imagen
ruta_imagen = 'Files/Videos/PelotaGolf.jpg'
imagen = cv2.imread(ruta_imagen)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Convertir a espacio de color HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    def nada(x):
        pass

    # Crear una ventana
    cv2.namedWindow('imagen')

    # Crear trackbars para cambiar los valores de color
    cv2.createTrackbar('HMin', 'imagen', 0, 179, nada)
    cv2.createTrackbar('SMin', 'imagen', 0, 255, nada)
    cv2.createTrackbar('VMin', 'imagen', 0, 255, nada)
    cv2.createTrackbar('HMax', 'imagen', 0, 179, nada)
    cv2.createTrackbar('SMax', 'imagen', 0, 255, nada)
    cv2.createTrackbar('VMax', 'imagen', 0, 255, nada)

    # Establecer valores por defecto para las trackbars de Max HSV
    cv2.setTrackbarPos('HMax', 'imagen', 179)
    cv2.setTrackbarPos('SMax', 'imagen', 255)
    cv2.setTrackbarPos('VMax', 'imagen', 255)

    while True:
        # Obtener posiciones actuales de todas las trackbars
        h_min = cv2.getTrackbarPos('HMin', 'imagen')
        s_min = cv2.getTrackbarPos('SMin', 'imagen')
        v_min = cv2.getTrackbarPos('VMin', 'imagen')
        h_max = cv2.getTrackbarPos('HMax', 'imagen')
        s_max = cv2.getTrackbarPos('SMax', 'imagen')
        v_max = cv2.getTrackbarPos('VMax', 'imagen')
        
        # Establecer valores mínimos y máximos de HSV para mostrar
        inferior = np.array([h_min, s_min, v_min])
        superior = np.array([h_max, s_max, v_max])
        
        # Crear una máscara basada en los valores HSV actuales
        mascara = cv2.inRange(imagen_hsv, inferior, superior)
        resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)
        
        # Mostrar la imagen de resultado
        cv2.imshow('imagen', resultado)
        
        # Romper el bucle cuando se presione 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

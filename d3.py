import cv2
import numpy as np
from matplotlib import pyplot as plt


def inputImg():
    while True:
        ruta = input("\n\tIngrese ruta de la imagen: ")
        img = cv2.imread(ruta, 0)

        if img is not None:
            return img
        else:
            print("\n\tNo se pudo abrir la imagen.\n")

def obtener_limites(proyeccion, centro):

    umbral = proyeccion[centro] * 0.95

    # Se toma el 95% del valor máximo del pico
    # para determinar automáticamente el ancho
    # de cada banda de ruido.

    izquierda = centro

    while izquierda > 0 and proyeccion[izquierda] > umbral:
        izquierda -= 1

    derecha = centro

    while derecha < len(proyeccion)-1 and proyeccion[derecha] > umbral:
        derecha += 1

    return izquierda, derecha

# ==========================
# PROGRAMA PRINCIPAL
# ==========================

while True:

    print("\n\n************************************************************")
    print("\n\t\tPROCESAMIENTO DIGITAL DE IMÁGENES")
    print("\n\t\t    TRANSFORMADA DE FOURIER")
    print("\n************************************************************")

    img = inputImg()

    cv2.imshow("Imagen Original", img)
    cv2.waitKey(0)

    img_float32 = np.float32(img)

    # ==========================
    #  Transformada de Fourier
    # ==========================

    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    magnitude = cv2.magnitude(dft_shift[:, :, 0],
                              dft_shift[:, :, 1])

    magnitude_spectrum = 20 * np.log(magnitude + 1)

    # ==========================
    # EPreparar el espectro
    # ==========================

    spec = cv2.normalize(magnitude_spectrum,
                         None,
                         0,
                         255,
                         cv2.NORM_MINMAX)

    spec = np.uint8(spec)

    # ==========================
    # SUAVIZAR EL NEGATIVO
    # ==========================

    suavizada = cv2.GaussianBlur(spec, (5,5), 0)
    
    # ==========================
    # TAMAÑO DE LA IMAGEN
    # ==========================

    rows, cols = suavizada.shape

    crow = rows // 2
    ccol = cols // 2

    # ==========================
    #  PROYECCIÓN POR COLUMNAS
    # ==========================

    proyeccion = np.sum(suavizada.astype(np.float32), axis=0)

    # Eliminar la componente DC de la búsqueda
    proyeccion[ccol-20:ccol+20] = np.max(proyeccion)
    plt.figure(figsize=(10,4))
    plt.plot(proyeccion)
    plt.grid(True)
    plt.title("Proyección por columnas")
    plt.show()

    # ==========================
    # DETECCIÓN DE LAS DOS BANDAS
    # ==========================

    # Mitad izquierda
    izquierda = proyeccion[:ccol-20]

    # Mitad derecha
    derecha = proyeccion[ccol+20:]

    # Centro de la banda izquierda
    x1 = np.argmax(izquierda)

    # Centro de la banda derecha
    x2 = np.argmax(derecha) + ccol + 20

    print("Centros:", x1, x2)

    # Obtener inicio y fin de cada banda
    inicio1, fin1 = obtener_limites(proyeccion, x1)
    inicio2, fin2 = obtener_limites(proyeccion, x2)

    print("Banda izquierda:", inicio1, fin1)
    print("Banda derecha :", inicio2, fin2)


    deteccion = cv2.cvtColor(spec, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(deteccion,
              (inicio1, 0),
              (fin1, rows-1),
              (0,0,255),
              2)

    cv2.rectangle(deteccion,
                (inicio2, 0),
                (fin2, rows-1),
                (0,0,255),
                2)

    cv2.imshow("Bandas detectadas", deteccion)
    cv2.waitKey(0)
    cv2.destroyWindow("Bandas detectadas")

    # ==========================
    # CREAR MÁSCARA BAND REJECT
    # ==========================

    mask = np.ones((rows, cols, 2), np.uint8)

    margen = 2

    mask[:, max(0, inicio1-margen):min(cols, fin1+margen)] = 0
    mask[:, max(0, inicio2-margen):min(cols, fin2+margen)] = 0

    # Conservar la componente DC
    mask[crow-8:crow+8,
        ccol-8:ccol+8] = 1

    cv2.imshow("Mascara", mask[:,:,0]*255)
    cv2.waitKey(0)
    cv2.destroyWindow("Mascara")

    # ==========================
    # ESPECTRO FILTRADO
    # ==========================

    fshift = dft_shift * mask

    magnitude_filtered = cv2.magnitude(fshift[:, :, 0],
                                       fshift[:, :, 1])

    magnitude_filtered = 20 * np.log(magnitude_filtered + 1)

    # ==========================
    # TRANSFORMADA INVERSA
    # ==========================

    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)

    img_back = cv2.magnitude(img_back[:, :, 0],
                             img_back[:, :, 1])

    img_back = cv2.normalize(img_back,
                             None,
                             0,
                             255,
                             cv2.NORM_MINMAX)

    img_back = np.uint8(img_back)

    # ==========================
    # ESPECTRO DE LA IMAGEN RESTAURADA
    # ==========================

    img_back_float = np.float32(img_back)

    dft_restored = cv2.dft(img_back_float, flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_restored_shift = np.fft.fftshift(dft_restored)

    magnitude_restored = cv2.magnitude(
        dft_restored_shift[:, :, 0],
        dft_restored_shift[:, :, 1]
    )

    magnitude_restored = 20 * np.log(magnitude_restored + 1)

    # ==========================
    # RESULTADOS
    # ==========================

    plt.figure(figsize=(10,6))

    plt.subplot(241)
    plt.imshow(img,cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(242)
    plt.imshow(magnitude_spectrum,cmap='gray')
    plt.title("Espectro")
    plt.axis("off")

    plt.subplot(243)
    plt.imshow(mask[:,:,0],cmap='gray')
    plt.title("Band Reject")
    plt.axis("off")

    plt.subplot(244)
    plt.imshow(magnitude_filtered,cmap='gray')
    plt.title("Espectro Filtrado")
    plt.axis("off")

    plt.subplot(245)
    plt.imshow(img_back,cmap='gray')
    plt.title("Imagen Restaurada")
    plt.axis("off")

    plt.subplot(246)
    plt.imshow(magnitude_restored,cmap='gray')
    plt.title("Espectro Restaurado")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    op = input("\n¿Procesar otra imagen? (s/n): ")

    if op.lower() != "s":
        break

cv2.destroyAllWindows()

 # images_deber_lab7/deber#1_ruido_periodico.jpg

 # "Se calcula la proyección vertical del espectro suavizado, ignorando la componente DC. 
 # Luego se buscan los dos máximos correspondientes a las bandas de ruido y, usando un umbral del 95 % de la 
 # amplitud del pico, se determinan automáticamente el inicio y el fin de cada banda para construir el 
 # filtro Band Reject."
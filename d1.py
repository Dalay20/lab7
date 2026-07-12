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


# ==========================
# VARIABLES GLOBALES
# ==========================

points = []
spec_img = None


# ==========================
# EVENTO DEL MOUSE
# ==========================

def mouse_click(event, x, y, flags, param):

    global points, spec_img

    if event == cv2.EVENT_LBUTTONDOWN:

        # Solo aceptar dos clics
        if len(points) >= 2:
            return

        points.append((x, y))

        # Dibujar una línea vertical
        cv2.line(spec_img,
                 (x, 0),
                 (x, spec_img.shape[0]),
                 255,
                 1)

        # Dibujar el punto
        cv2.circle(spec_img,
                   (x, y),
                   5,
                   255,
                   -1)

        cv2.imshow("Espectro", spec_img)


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
    # DFT
    # ==========================

    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    magnitude = cv2.magnitude(dft_shift[:, :, 0],
                              dft_shift[:, :, 1])

    magnitude_spectrum = 20 * np.log(magnitude + 1)

    # ==========================
    # ESPECTRO PARA SELECCIONAR
    # ==========================

    spec = cv2.normalize(magnitude_spectrum,
                         None,
                         0,
                         255,
                         cv2.NORM_MINMAX)

    spec = np.uint8(spec)

    spec_img = spec.copy()

    points = []

    cv2.imshow("Espectro", spec_img)

    cv2.setMouseCallback("Espectro", mouse_click)

    print("\nSeleccione las DOS bandas de ruido.")
    print("Después presione cualquier tecla.")

    cv2.waitKey(0)

    cv2.destroyWindow("Espectro")

    # ==========================
    # CREAR MÁSCARA
    # ==========================

    rows, cols = img.shape

    crow = rows // 2
    ccol = cols // 2

    mask = np.ones((rows, cols, 2), np.uint8)

    ancho = 4

    for (x, y) in points:

        x1 = max(0, x - ancho)
        x2 = min(cols, x + ancho)

        mask[:, x1:x2] = 0

    # Mantener la frecuencia DC
    mask[crow-8:crow+8,
         ccol-8:ccol+8] = 1

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

    plt.figure(figsize=(12,8))

    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(232)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Espectro")
    plt.axis("off")

    plt.subplot(233)
    plt.imshow(mask[:, :, 0], cmap='gray')
    plt.title("Máscara")
    plt.axis("off")

    plt.subplot(234)
    plt.imshow(magnitude_filtered, cmap='gray')
    plt.title("Espectro Filtrado")
    plt.axis("off")

    plt.subplot(235)
    plt.imshow(img_back, cmap='gray')
    plt.title("Imagen Restaurada")
    plt.axis("off")

    plt.subplot(236)
    plt.imshow(magnitude_restored, cmap='gray')
    plt.title("Espectro Restaurado")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    op = input("\n¿Procesar otra imagen? (s/n): ")

    if op.lower() != "s":
        break

cv2.destroyAllWindows()

 # images_deber_lab7/deber#1_ruido_periodico.jpg

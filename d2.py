import cv2
import numpy as np
from matplotlib import pyplot as plt

# Lista de imágenes
imagenes = [
    ("Original", "images_deber_lab7/deber#2_img_original.png"),
    ("Suavizada", "images_deber_lab7/deber#2_img_suavizada.png"),
    ("Con Ruido", "images_deber_lab7/deber#2_img_ruido.png"),
    ("Trasladada", "images_deber_lab7/deber#2_img_trasladada.png"),
    ("Perfilada", "images_deber_lab7/deber#2_img_perfilada.png"),
    ("Rotada", "images_deber_lab7/deber#2_img_rotada.png"),
    ("Reducida", "images_deber_lab7/deber#2_img_reducida.png")
]

def mostrar_grupo(lista_imagenes, titulo_figura):
    fig, axs = plt.subplots(len(lista_imagenes), 2, figsize=(10, 4*len(lista_imagenes)))
    fig.suptitle(titulo_figura, fontsize=16)

    # Cuando solo hay una imagen, axs no es una matriz
    if len(lista_imagenes) == 1:
        axs = [axs]

    for i, (titulo, archivo) in enumerate(lista_imagenes):

        img = cv2.imread(archivo, 0)

        if img is None:
            print(f"No se pudo abrir {archivo}")
            continue

        img_float32 = np.float32(img)

        # Transformada de Fourier
        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude = cv2.magnitude(dft_shift[:, :, 0],
                                  dft_shift[:, :, 1])

        magnitude = 20 * np.log(magnitude + 1)

        # Imagen
        axs[i][0].imshow(img, cmap="gray")
        axs[i][0].set_title(titulo)
        axs[i][0].axis("off")

        # Espectro
        axs[i][1].imshow(magnitude, cmap="gray")
        axs[i][1].set_title("Espectro de Fourier")
        axs[i][1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Primer grupo
mostrar_grupo(imagenes[:2], "Grupo 1")

# Segundo grupo
mostrar_grupo(imagenes[2:4], "Grupo 2")

# Tercer grupo
mostrar_grupo(imagenes[4:7], "Grupo 3")
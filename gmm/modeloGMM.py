import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from PIL import Image
import os

# Crear carpeta para guardar los clusters
carpeta_clusters = 'clusters'
if not os.path.exists(carpeta_clusters):
    os.makedirs(carpeta_clusters)

# Cargar y Pre-procesar la Imagen
imagen = Image.open('pirataverdu.png')
imagen_rgb = np.array(imagen.convert('RGB'))

# Reformatear: de (alto, ancho, canales) a (alto * ancho, canales) para el GMM
alto, ancho, canales = imagen_rgb.shape
datos_pixeles = imagen_rgb.reshape((-1, canales))

# Ajustar el Modelo de Mezcla Gaussiana
n_componentes = 20
gmm = GaussianMixture(n_components=n_componentes, random_state=0)
gmm.fit(datos_pixeles)

# Predecir y Reconstruir
etiquetas = gmm.predict(datos_pixeles)

# Asignar cada etiqueta al centroide del cluster
colores_cluster = gmm.means_.astype(int)  # colores promedio de cada cluster
imagen_segmentada = colores_cluster[etiquetas].reshape((alto, ancho, 3))

# Crear y guardar cada cluster individual en la carpeta
for i in range(n_componentes):
    # Crear una máscara para el cluster actual
    mascara_cluster = (etiquetas == i).reshape(alto, ancho)
    
    # Crear una imagen RGBA (con canal alpha) para el cluster
    imagen_cluster = np.zeros((alto, ancho, 4), dtype=np.uint8)
    
    # Para los píxeles que pertenecen al cluster, usar el color original
    # y establecer alpha = 255 (opaco)
    imagen_cluster[mascara_cluster, :3] = imagen_rgb[mascara_cluster]
    imagen_cluster[mascara_cluster, 3] = 255  # Canal alpha
    
    # Para los píxeles que NO pertenecen al cluster, alpha = 0 (transparente)
    imagen_cluster[~mascara_cluster, 3] = 0
    
    # Convertir a imagen PIL y guardar en la carpeta
    img_pil = Image.fromarray(imagen_cluster, 'RGBA')
    ruta_cluster = os.path.join(carpeta_clusters, f'cluster_{i:02d}.png')
    img_pil.save(ruta_cluster)
    
    print(f'Cluster {i} guardado: {mascara_cluster.sum()} píxeles')

# Crear imagen fusionada de todos los clusters
imagen_fusionada = np.zeros((alto, ancho, 4), dtype=np.uint8)

# Combinar todos los clusters
for i in range(n_componentes):
    # Cargar cada cluster desde la carpeta
    ruta_cluster = os.path.join(carpeta_clusters, f'cluster_{i:02d}.png')
    cluster_img = np.array(Image.open(ruta_cluster))
    
    # Para cada píxel, si no es transparente en el cluster actual, agregarlo a la fusión
    mascara_opaca = cluster_img[:, :, 3] > 0
    imagen_fusionada[mascara_opaca] = cluster_img[mascara_opaca]

# Guardar la imagen fusionada en la carpeta
ruta_fusion = os.path.join(carpeta_clusters, 'fusion_clusters.png')
Image.fromarray(imagen_fusionada, 'RGBA').save(ruta_fusion)

# Visualización
plt.figure(figsize=(20, 5))

# Imagen original
plt.subplot(1, 4, 1)
plt.imshow(imagen_rgb)
plt.title('Imagen Original')
plt.axis('off')

# Imagen segmentada con GMM
plt.subplot(1, 4, 2)
plt.imshow(imagen_segmentada)
plt.title('Imagen Segmentada con GMM (RGB)')
plt.axis('off')

# Ejemplo de cluster individual
plt.subplot(1, 4, 3)
ruta_ejemplo = os.path.join(carpeta_clusters, 'cluster_00.png')
cluster_ejemplo = np.array(Image.open(ruta_ejemplo))
plt.imshow(cluster_ejemplo)
plt.title('Ejemplo: Cluster 0 (con transparencia)')
plt.axis('off')

# Imagen fusionada de todos los clusters
plt.subplot(1, 4, 4)
plt.imshow(imagen_fusionada)
plt.title('Fusión de Todos los Clusters')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"\nSe han guardado {n_componentes} clusters individuales en la carpeta '{carpeta_clusters}'.")
print(f"Se ha guardado 'fusion_clusters.png' en la carpeta '{carpeta_clusters}' con la combinación de todos los clusters.")
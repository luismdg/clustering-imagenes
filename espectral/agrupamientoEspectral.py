import numpy as np
import cv2
from sklearn.cluster import SpectralClustering
from PIL import Image
import os
import time
from sklearn.neighbors import kneighbors_graph

def procesar_verduras_espectral_corregido(ruta_imagen, n_clusters=21):
    start_time = time.time()
    
    # Verificar si la imagen existe
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encuentra el archivo {ruta_imagen}")
        return
    
    # Crear carpeta de salida si no existe
    carpeta_salida = 'espectral'
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Cargar imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return
    
    print(f"Imagen cargada correctamente. Dimensiones: {img.shape}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img_rgb.shape[:2]
    
    # Reducir tamaño pero mantener más detalle para 21 clusters
    scale_factor = min(400 / original_shape[1], 400 / original_shape[0])
    new_width = int(original_shape[1] * scale_factor)
    new_height = int(original_shape[0] * scale_factor)
    img_small = cv2.resize(img_rgb, (new_width, new_height))
    small_shape = img_small.shape[:2]
    print(f"Imagen redimensionada a: {small_shape}")
    
    # Aplicar filtro para suavizar
    img_filtered = cv2.medianBlur(img_small, 5)
    
    # Preparar datos para clustering - usar LAB para mejor separación de colores
    img_lab = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2LAB)
    h, w = small_shape
    
    # Estrategia: usar una cuadrícula de superpíxeles + muestreo aleatorio
    grid_size = 15
    y_coords = np.arange(0, h, grid_size)
    x_coords = np.arange(0, w, grid_size)
    
    muestras_posiciones = []
    muestras_colores = []
    
    # Muestrear de la cuadrícula
    for y in y_coords:
        for x in x_coords:
            muestras_posiciones.append([y / h, x / w])  # Posición normalizada
            muestras_colores.append(img_lab[y, x] / 255.0)  # Color LAB normalizado
    
    # Agregar muestras aleatorias adicionales para asegurar suficientes puntos
    n_muestras_extra = max(0, 2000 - len(muestras_posiciones))
    if n_muestras_extra > 0:
        for _ in range(n_muestras_extra):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            muestras_posiciones.append([y / h, x / w])
            muestras_colores.append(img_lab[y, x] / 255.0)
    
    # Combinar características
    muestras_posiciones = np.array(muestras_posiciones)
    muestras_colores = np.array(muestras_colores)
    muestras = np.hstack([muestras_posiciones, muestras_colores])
    
    print(f"Total de muestras: {len(muestras)}")
    
    # Aplicar Spectral Clustering
    print("Aplicando Spectral Clustering...")
    
    # Construir gráfico de vecinos más cercanos
    n_neighbors = min(15, len(muestras) - 1)
    connectivity = kneighbors_graph(
        muestras, 
        n_neighbors=n_neighbors, 
        include_self=False
    )
    
    # Spectral Clustering con matriz de afinidad precomputada
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='discretize',
        n_init=5
    )
    
    # Convertir a matriz de afinidad simétrica
    affinity_matrix = 0.5 * (connectivity + connectivity.T)
    affinity_matrix = affinity_matrix.toarray()
    
    etiquetas_muestras = clustering.fit_predict(affinity_matrix)
    
    # Asignar etiquetas a todos los píxeles usando interpolación
    from sklearn.neighbors import KNeighborsClassifier
    
    print("Interpolando etiquetas a todos los píxeles...")
    
    # Preparar todos los píxeles de la imagen pequeña
    pixeles_completos = []
    for y in range(h):
        for x in range(w):
            pos_normalizada = [y / h, x / w]
            color_normalizado = img_lab[y, x] / 255.0
            pixeles_completos.append(np.concatenate([pos_normalizada, color_normalizado]))
    
    pixeles_completos = np.array(pixeles_completos)
    
    # Usar KNN para asignar etiquetas basado en las muestras
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(muestras, etiquetas_muestras)
    
    # Predecir en lotes para evitar problemas de memoria
    etiquetas_completas = np.zeros(h * w, dtype=np.int32)
    batch_size = 5000
    for i in range(0, h * w, batch_size):
        end_idx = min(i + batch_size, h * w)
        etiquetas_completas[i:end_idx] = knn.predict(pixeles_completos[i:end_idx])
    
    etiquetas_reshape = etiquetas_completas.reshape(small_shape)
    
    # Crear y guardar imagen del clustering
    print("Generando imagen de resultado del clustering...")
    resultado_clustering = np.zeros_like(img_small)
    # Generar colores distintivos
    colores = []
    for i in range(n_clusters):
        # Generar colores más distintivos usando HSV y convertir a RGB
        hue = (i * 180 // n_clusters)  # Distribuir en el círculo de color
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
        colores.append(color_rgb)
    colores = np.array(colores)
    
    for i in range(n_clusters):
        mascara = etiquetas_reshape == i
        resultado_clustering[mascara] = colores[i]
    
    # Guardar imagen del clustering
    ruta_resultado = os.path.join(carpeta_salida, 'resultado_espectral_corregido.png')
    Image.fromarray(resultado_clustering).save(ruta_resultado)
    
    # Crear carpeta para máscaras individuales
    carpeta_mascaras = os.path.join(carpeta_salida, 'mascaras_verduras_corregido')
    os.makedirs(carpeta_mascaras, exist_ok=True)
    
    # Guardar cada máscara individual
    print("Guardando máscaras individuales...")
    for i in range(n_clusters):
        mascara_small = (etiquetas_reshape == i).astype(np.uint8) * 255
        
        # Redimensionar máscara al tamaño original
        mascara_original = cv2.resize(mascara_small, (original_shape[1], original_shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((5, 5), np.uint8)
        mascara_original = cv2.morphologyEx(mascara_original, cv2.MORPH_CLOSE, kernel)
        mascara_original = cv2.morphologyEx(mascara_original, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos y mantener solo los significativos
        contours, _ = cv2.findContours(mascara_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area_total = original_shape[0] * original_shape[1]
            # Filtrar contornos por área (al menos 0.5% del área total)
            contornos_filtrados = [cnt for cnt in contours if cv2.contourArea(cnt) > area_total * 0.005]
            
            if contornos_filtrados:
                mascara_limpia = np.zeros_like(mascara_original)
                cv2.fillPoly(mascara_limpia, contornos_filtrados, 255)
                mascara_original = mascara_limpia
        
        # Crear imagen RGBA con fondo transparente
        rgba = np.zeros((*original_shape, 4), dtype=np.uint8)
        rgba[mascara_original == 255] = [0, 0, 0, 255]  # Negro opaco para la verdura
        rgba[mascara_original == 0] = [0, 0, 0, 0]      # Transparente para el fondo
        
        # Guardar con PIL
        ruta_mascara = os.path.join(carpeta_mascaras, f'vegetal_{i+1:02d}.png')
        Image.fromarray(rgba, 'RGBA').save(ruta_mascara)
        print(f"Guardada: {ruta_mascara}")
    
    end_time = time.time()
    print(f"Proceso completado en {end_time - start_time:.2f} segundos!")
    print(f"Imagen de clustering guardada en: {ruta_resultado}")
    print(f"Máscaras individuales guardadas en: {carpeta_mascaras}")
    
    return etiquetas_reshape

# Versión alternativa más simple pero funcional
def procesar_verduras_espectral_simple(ruta_imagen, n_clusters=21):
    start_time = time.time()
    
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encuentra el archivo {ruta_imagen}")
        return
    
    carpeta_salida = 'espectral'
    os.makedirs(carpeta_salida, exist_ok=True)
    
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return
    
    print(f"Imagen cargada correctamente. Dimensiones: {img.shape}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img_rgb.shape[:2]
    
    # Reducir tamaño para hacerlo manejable
    scale_factor = min(300 / original_shape[1], 300 / original_shape[0])
    new_width = int(original_shape[1] * scale_factor)
    new_height = int(original_shape[0] * scale_factor)
    img_small = cv2.resize(img_rgb, (new_width, new_height))
    small_shape = img_small.shape[:2]
    print(f"Imagen redimensionada a: {small_shape}")
    
    # Convertir a LAB para mejor clustering de color
    img_lab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)
    h, w = small_shape
    
    # Crear dataset de muestras (posición + color)
    n_muestras = min(1500, h * w)
    indices = np.random.choice(h * w, n_muestras, replace=False)
    
    muestras = []
    for idx in indices:
        y = idx // w
        x = idx % w
        # Características: posición normalizada + color LAB normalizado
        caracteristicas = [
            y / h, x / w,  # Posición
            img_lab[y, x, 0] / 255.0,  # L
            img_lab[y, x, 1] / 255.0,  # A
            img_lab[y, x, 2] / 255.0   # B
        ]
        muestras.append(caracteristicas)
    
    muestras = np.array(muestras)
    print(f"Muestras para clustering: {len(muestras)}")
    
    # Spectral Clustering directo
    print("Aplicando Spectral Clustering...")
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='rbf',
        gamma=0.1,
        random_state=42,
        assign_labels='kmeans',
        n_init=3
    )
    
    etiquetas_muestras = clustering.fit_predict(muestras)
    
    # Interpolar a toda la imagen
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(muestras, etiquetas_muestras)
    
    # Predecir para todos los píxeles
    print("Interpolando etiquetas...")
    pixeles_completos = []
    for y in range(h):
        for x in range(w):
            pixeles_completos.append([y/h, x/w, 
                                    img_lab[y, x, 0]/255.0, 
                                    img_lab[y, x, 1]/255.0, 
                                    img_lab[y, x, 2]/255.0])
    
    pixeles_completos = np.array(pixeles_completos)
    etiquetas_completas = knn.predict(pixeles_completos)
    etiquetas_reshape = etiquetas_completas.reshape(small_shape)
    
    # Resto del proceso igual...
    resultado_clustering = np.zeros_like(img_small)
    colores = np.random.randint(0, 255, size=(n_clusters, 3))
    
    for i in range(n_clusters):
        mascara = etiquetas_reshape == i
        resultado_clustering[mascara] = colores[i]
    
    ruta_resultado = os.path.join(carpeta_salida, 'resultado_espectral_simple.png')
    Image.fromarray(resultado_clustering).save(ruta_resultado)
    
    carpeta_mascaras = os.path.join(carpeta_salida, 'mascaras_verduras_simple')
    os.makedirs(carpeta_mascaras, exist_ok=True)
    
    print("Guardando máscaras individuales...")
    for i in range(n_clusters):
        mascara_small = (etiquetas_reshape == i).astype(np.uint8) * 255
        mascara_original = cv2.resize(mascara_small, (original_shape[1], original_shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        kernel = np.ones((5, 5), np.uint8)
        mascara_original = cv2.morphologyEx(mascara_original, cv2.MORPH_CLOSE, kernel)
        mascara_original = cv2.morphologyEx(mascara_original, cv2.MORPH_OPEN, kernel)
        
        rgba = np.zeros((*original_shape, 4), dtype=np.uint8)
        rgba[mascara_original == 255] = [0, 0, 0, 255]
        rgba[mascara_original == 0] = [0, 0, 0, 0]
        
        ruta_mascara = os.path.join(carpeta_mascaras, f'vegetal_{i+1:02d}.png')
        Image.fromarray(rgba, 'RGBA').save(ruta_mascara)
        print(f"Guardada: {ruta_mascara}")
    
    end_time = time.time()
    print(f"Proceso completado en {end_time - start_time:.2f} segundos!")
    print(f"Resultado en: {ruta_resultado}")
    print(f"Máscaras en: {carpeta_mascaras}")

if __name__ == "__main__":
    ruta_imagen = 'pirataverdu.png'
    
    if not os.path.exists(ruta_imagen):
        print("No se encontró la imagen pirataverdu.png")
        print("Directorio actual:", os.getcwd())
        print("Archivos en el directorio actual:")
        for item in os.listdir('.'):
            print(f"  {item}")
    else:
        print(f"Imagen encontrada en: {ruta_imagen}")
        
        n_clusters = 21
        
        print("=" * 60)
        print("EJECUTANDO VERSIÓN CORREGIDA")
        print("=" * 60)
        procesar_verduras_espectral_corregido(ruta_imagen, n_clusters)
        
        print("\n" + "=" * 60)
        print("EJECUTANDO VERSIÓN SIMPLE")
        print("=" * 60)
        procesar_verduras_espectral_simple(ruta_imagen, n_clusters)
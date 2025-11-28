import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from PIL import Image
import os
import argparse

def main():
    # Obtener la ruta del directorio actual del script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description='Segmentación de imagen usando GMM')
    parser.add_argument('--image_path', type=str, default='pirataverdu.png', 
                       help='Ruta de la imagen a segmentar')
    parser.add_argument('--n_components', type=int, default=20,
                       help='Número de clusters para GMM')
    parser.add_argument('--output_dir', type=str, default='clusters',
                       help='Directorio de salida para los clusters')
    
    args = parser.parse_args()
    
    # Construir rutas absolutas
    parent_dir = os.path.dirname(current_dir)
    image_path = os.path.join(parent_dir, args.image_path)
    output_dir = os.path.join(current_dir, args.output_dir)
    
    # Crear carpeta para guardar los clusters
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio '{output_dir}' creado.")
    
    # Cargar y verificar la imagen
    try:
        imagen = Image.open(image_path)
        print(f"Imagen cargada: {image_path}")
        print(f"Tamaño original: {imagen.size}")
        print(f"Modo: {imagen.mode}")
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar la imagen en {image_path}")
        return
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return

    # Convertir a RGB si es necesario
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
        print("Imagen convertida a RGB")
    
    # Redimensionar a 1920x1080 si es necesario
    target_size = (1920, 1080)
    if imagen.size != target_size:
        print(f"Redimensionando imagen de {imagen.size} a {target_size}")
        imagen = imagen.resize(target_size, Image.Resampling.LANCZOS)
    
    imagen_rgb = np.array(imagen)

    # Reformatear: de (alto, ancho, canales) a (alto * ancho, canales) para el GMM
    alto, ancho, canales = imagen_rgb.shape
    datos_pixeles = imagen_rgb.reshape((-1, canales))
    
    print(f"Dimensiones de la imagen: {alto}x{ancho}")
    print(f"Datos para GMM: {datos_pixeles.shape}")

    # Ajustar el Modelo de Mezcla Gaussiana
    print(f"\nAjustando GMM con {args.n_components} componentes...")
    gmm = GaussianMixture(n_components=args.n_components, random_state=0)
    gmm.fit(datos_pixeles)
    print("GMM ajustado correctamente")

    # Predecir y Reconstruir
    print("Prediciendo clusters...")
    etiquetas = gmm.predict(datos_pixeles)

    # Asignar cada etiqueta al centroide del cluster
    colores_cluster = gmm.means_.astype(int)  # colores promedio de cada cluster
    imagen_segmentada = colores_cluster[etiquetas].reshape((alto, ancho, 3))

    # Crear y guardar cada cluster individual en la carpeta
    print("\nGuardando clusters individuales...")
    for i in range(args.n_components):
        # Crear una máscara para el cluster actual
        mascara_cluster = (etiquetas == i).reshape(alto, ancho)
        
        # --- Crear PNG con fondo transparente y figura negra ---
        imagen_cluster = np.zeros((alto, ancho, 4), dtype=np.uint8)
        
        # Donde hay cluster: negro sólido (RGB = 0,0,0) y alpha = 255
        # Donde no hay cluster: transparente (alpha = 0)
        imagen_cluster[mascara_cluster, 3] = 255  # Alpha = 255 donde hay objeto
        
        # Los canales RGB ya están en 0 por el np.zeros, así que queda negro
        
        # Convertir a imagen PIL y guardar en la carpeta
        img_pil = Image.fromarray(imagen_cluster, 'RGBA')
        ruta_cluster = os.path.join(output_dir, f'cluster_{i:02d}.png')
        img_pil.save(ruta_cluster)
        
        pixeles_cluster = mascara_cluster.sum()
        print(f'Cluster {i:02d} guardado: {pixeles_cluster:,} píxeles ({pixeles_cluster/(alto*ancho)*100:.1f}%)')

    # Crear imagen fusionada de todos los clusters (en negro también)
    print("\nCreando imagen fusionada...")
    imagen_fusionada = np.zeros((alto, ancho, 4), dtype=np.uint8)

    # Combinar todos los clusters
    for i in range(args.n_components):
        # Cargar cada cluster desde la carpeta
        ruta_cluster = os.path.join(output_dir, f'cluster_{i:02d}.png')
        cluster_img = np.array(Image.open(ruta_cluster))
        
        # Para cada píxel, si no es transparente en el cluster actual, agregarlo a la fusión
        mascara_opaca = cluster_img[:, :, 3] > 0
        imagen_fusionada[mascara_opaca, 3] = 255  # Alpha = 255 donde hay cualquier cluster

    # Guardar la imagen fusionada en la carpeta
    ruta_fusion = os.path.join(output_dir, 'fusion_clusters.png')
    Image.fromarray(imagen_fusionada, 'RGBA').save(ruta_fusion)
    print("Imagen fusionada guardada")

    # Visualización
    print("\nGenerando visualización...")
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

    # Ejemplo de cluster individual (ahora en negro)
    plt.subplot(1, 4, 3)
    ruta_ejemplo = os.path.join(output_dir, 'cluster_00.png')
    cluster_ejemplo = np.array(Image.open(ruta_ejemplo))
    # Para visualizar clusters negros, mostramos el canal alpha como escala de grises
    plt.imshow(cluster_ejemplo[:, :, 3], cmap='gray')
    plt.title('Ejemplo: Cluster 0 (negro con transparencia)')
    plt.axis('off')

    # Imagen fusionada de todos los clusters
    plt.subplot(1, 4, 4)
    plt.imshow(imagen_fusionada[:, :, 3], cmap='gray')
    plt.title('Fusión de Todos los Clusters')
    plt.axis('off')

    plt.tight_layout()
    
    # Guardar la figura de visualización
    ruta_visualizacion = os.path.join(output_dir, 'visualizacion.png')
    plt.savefig(ruta_visualizacion, dpi=150, bbox_inches='tight')
    print(f"Visualización guardada en: {ruta_visualizacion}")
    
    # Mostrar la figura
    plt.show()

    print(f"\n" + "="*50)
    print(f"PROCESO COMPLETADO")
    print(f"="*50)
    print(f"Se han guardado {args.n_components} clusters individuales en la carpeta '{output_dir}'")
    print(f"Dimensiones: {ancho}x{alto}")
    print(f"Archivos generados:")
    print(f"  - cluster_00.png a cluster_{args.n_components-1:02d}.png (negro con transparencia)")
    print(f"  - fusion_clusters.png")
    print(f"  - visualizacion.png")
    print(f"Ruta completa: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
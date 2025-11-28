# comparacion.py - Versión con DEPURACIÓN
import os
import numpy as np
from PIL import Image
import glob

def debug_mascara(mascara, nombre):
    """Función para depurar máscaras"""
    print(f"   🔍 DEBUG {nombre}:")
    print(f"      - Shape: {mascara.shape}")
    print(f"      - Tipo: {mascara.dtype}")
    print(f"      - Valores únicos: {np.unique(mascara)}")
    print(f"      - Píxeles True: {np.sum(mascara):,}")
    print(f"      - Porcentaje True: {np.sum(mascara) / mascara.size * 100:.2f}%")

def calcular_metricas(gt_mask, pred_mask):
    """
    Calcula métricas de comparación entre máscaras
    gt_mask: máscara ground truth (verdurinipiratini)
    pred_mask: máscara predicción (clusters)
    """
    # Asegurar que son booleanas
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    
    # Calcular intersección y unión
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    
    # IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0.0
    
    # Precision: verdadero positivos / (verdadero positivos + falsos positivos)
    true_positives = intersection
    false_positives = np.logical_and(pred, np.logical_not(gt)).sum()
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # Recall: verdadero positivos / (verdadero positivos + falsos negativos)
    false_negatives = np.logical_and(gt, np.logical_not(pred)).sum()
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # F1-score: media armónica de precision y recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'intersection': intersection,
        'union': union
    }

def comparar_clusters_con_verdurinipiratini():
    """
    Compara imágenes de clusters con imágenes de verdurinipiratini
    Calcula métricas IoU, Precision, Recall y F1-score
    """
    # Obtener el directorio actual del script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuración de rutas
    clusters_dir = os.path.join(current_dir, 'clusters_renombrados')
    verdurinipiratini_dir = os.path.join(current_dir, '..', 'verdurinipiratini')
    output_dir = os.path.join(current_dir, 'comparacion_resultados')
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener lista de archivos PNG en verdurinipiratini
    archivos_verdurinipiratini = glob.glob(os.path.join(verdurinipiratini_dir, '*.png'))
    
    print(f"🔍 Encontrados {len(archivos_verdurinipiratini)} archivos en {verdurinipiratini_dir}")
    print("🔄 Iniciando comparación con DEPURACIÓN...")
    
    resultados = []
    
    for ruta_verdurinipiratini in archivos_verdurinipiratini:
        try:
            archivo = os.path.basename(ruta_verdurinipiratini)
            # Ruta correspondiente en clusters
            ruta_cluster = os.path.join(clusters_dir, archivo)
            
            # Verificar que existe el archivo correspondiente en clusters
            if not os.path.exists(ruta_cluster):
                print(f"⚠️  No se encontró {archivo} en clusters, saltando...")
                continue
            
            # Cargar imágenes con PIL
            img_verdurinipiratini = Image.open(ruta_verdurinipiratini)
            img_cluster = Image.open(ruta_cluster)
            
            print(f"\n📁 Procesando: {archivo}")
            print(f"   - Verdurinipiratini: {img_verdurinipiratini.size}, {img_verdurinipiratini.mode}")
            print(f"   - Cluster: {img_cluster.size}, {img_cluster.mode}")
            
            # DEPURACIÓN: Mostrar información de las imágenes originales
            verdurinipiratini_array = np.array(img_verdurinipiratini)
            cluster_array = np.array(img_cluster)
            
            print(f"   📊 Array Verdurinipiratini: shape {verdurinipiratini_array.shape}, dtype {verdurinipiratini_array.dtype}")
            print(f"   📊 Array Cluster: shape {cluster_array.shape}, dtype {cluster_array.dtype}")
            
            # Convertir verdurinipiratini a máscara binaria - MÚLTIPLES ESTRATEGIAS
            if img_verdurinipiratini.mode == 'RGBA':
                # Estrategia 1: Usar canal alpha
                mascara_verdurinipiratini_alpha = verdurinipiratini_array[:, :, 3] > 0
                # Estrategia 2: Usar cualquier canal que no sea transparente
                mascara_verdurinipiratini_rgb = np.any(verdurinipiratini_array[:, :, :3] > 0, axis=2)
                # Elegir la que tenga más píxeles True
                if mascara_verdurinipiratini_alpha.sum() > mascara_verdurinipiratini_rgb.sum():
                    mascara_verdurinipiratini = mascara_verdurinipiratini_alpha
                    print(f"   🎯 Usando canal ALPHA para verdurinipiratini")
                else:
                    mascara_verdurinipiratini = mascara_verdurinipiratini_rgb
                    print(f"   🎯 Usando canal RGB para verdurinipiratini")
            else:
                # Imagen sin alpha
                verdurinipiratini_gray = np.array(img_verdurinipiratini.convert('L'))
                mascara_verdurinipiratini = verdurinipiratini_gray > 0
            
            # Convertir cluster a máscara binaria
            if img_cluster.mode == 'RGBA':
                # Para clusters: usar canal alpha (debería ser la máscara)
                mascara_cluster = cluster_array[:, :, 3] > 0
                print(f"   🎯 Usando canal ALPHA para cluster")
            else:
                # Cluster sin alpha
                cluster_gray = np.array(img_cluster.convert('L'))
                mascara_cluster = cluster_gray > 0
            
            # DEPURACIÓN: Mostrar estadísticas de las máscaras
            debug_mascara(mascara_verdurinipiratini, "Verdurinipiratini")
            debug_mascara(mascara_cluster, "Cluster")
            
            # Redimensionar si es necesario para que coincidan los tamaños
            if mascara_cluster.shape != mascara_verdurinipiratini.shape:
                print(f"   🔄 Redimensionando cluster para coincidir tamaños...")
                img_cluster_resized = img_cluster.resize(img_verdurinipiratini.size, Image.NEAREST)
                cluster_array_resized = np.array(img_cluster_resized)
                
                if img_cluster_resized.mode == 'RGBA':
                    mascara_cluster = cluster_array_resized[:, :, 3] > 0
                else:
                    cluster_gray_resized = np.array(img_cluster_resized.convert('L'))
                    mascara_cluster = cluster_gray_resized > 0
                
                debug_mascara(mascara_cluster, "Cluster Redimensionado")
            
            # Calcular métricas
            metricas = calcular_metricas(mascara_verdurinipiratini, mascara_cluster)
            
            # DEPURACIÓN: Mostrar métricas detalladas
            print(f"   📈 MÉTRICAS:")
            print(f"      - Intersección: {metricas['intersection']:,}")
            print(f"      - Unión: {metricas['union']:,}")
            print(f"      - Verdaderos Positivos: {metricas['true_positives']:,}")
            print(f"      - Falsos Positivos: {metricas['false_positives']:,}")
            print(f"      - Falsos Negativos: {metricas['false_negatives']:,}")
            
            # Crear visualización de la comparación
            visualizacion = crear_visualizacion_comparacion(
                mascara_cluster, 
                mascara_verdurinipiratini
            )
            
            # Guardar visualización
            ruta_visualizacion = os.path.join(output_dir, f"vis_{archivo}")
            visualizacion.save(ruta_visualizacion)
            
            resultados.append({
                'archivo': archivo,
                'metricas': metricas,
                'pixeles_cluster': np.sum(mascara_cluster),
                'pixeles_verdurinipiratini': np.sum(mascara_verdurinipiratini)
            })
            
            print(f"✅ {archivo} → IoU = {metricas['iou']:.4f}")
            
        except Exception as e:
            print(f"❌ Error procesando {archivo}: {e}")
            import traceback
            traceback.print_exc()
    
    if not resultados:
        print("❌ No se encontraron archivos para comparar.")
        return
    
    # Generar reporte detallado
    generar_reporte_detallado(resultados, output_dir)
    print(f"\n🎉 Comparación completada. Resultados en: {output_dir}")

def crear_visualizacion_comparacion(mascara_cluster, mascara_verdurinipiratini):
    """
    Crea una visualización que muestra:
    - Verde: Verdaderos positivos (coinciden ambos)
    - Rojo: Falsos positivos (solo en cluster)
    - Azul: Falsos negativos (solo en verdurinipiratini)
    """
    # Crear imagen RGB
    visualizacion = np.zeros((*mascara_cluster.shape, 3), dtype=np.uint8)
    
    # Verdaderos positivos (intersección) - VERDE
    true_positives = np.logical_and(mascara_cluster, mascara_verdurinipiratini)
    visualizacion[true_positives] = [0, 255, 0]  # Verde
    
    # Falsos positivos (solo en cluster) - ROJO
    false_positives = np.logical_and(mascara_cluster, np.logical_not(mascara_verdurinipiratini))
    visualizacion[false_positives] = [255, 0, 0]  # Rojo
    
    # Falsos negativos (solo en verdurinipiratini) - AZUL
    false_negatives = np.logical_and(mascara_verdurinipiratini, np.logical_not(mascara_cluster))
    visualizacion[false_negatives] = [0, 0, 255]  # Azul
    
    return Image.fromarray(visualizacion)

def generar_reporte_detallado(resultados, output_dir):
    """Genera un reporte detallado con todas las métricas"""
    ruta_reporte = os.path.join(output_dir, "reporte_metricas_detallado.txt")
    
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("REPORTE DETALLADO - MÉTRICAS DE COMPARACIÓN\n")
        f.write("=" * 80 + "\n\n")
        
        # Estadísticas generales
        iou_promedio = np.mean([r['metricas']['iou'] for r in resultados])
        precision_promedio = np.mean([r['metricas']['precision'] for r in resultados])
        recall_promedio = np.mean([r['metricas']['recall'] for r in resultados])
        f1_promedio = np.mean([r['metricas']['f1'] for r in resultados])
        
        f.write(f"ESTADÍSTICAS GENERALES:\n")
        f.write(f"IoU Promedio: {iou_promedio:.4f}\n")
        f.write(f"Precision Promedio: {precision_promedio:.4f}\n")
        f.write(f"Recall Promedio: {recall_promedio:.4f}\n")
        f.write(f"F1-Score Promedio: {f1_promedio:.4f}\n")
        f.write(f"Archivos procesados: {len(resultados)}\n\n")
        
        f.write("DETALLE POR ARCHIVO:\n")
        f.write("=" * 80 + "\n")
        
        for res in resultados:
            m = res['metricas']
            f.write(f"Archivo: {res['archivo']}\n")
            f.write(f"  IoU: {m['iou']:.4f}\n")
            f.write(f"  Precision: {m['precision']:.4f}\n")
            f.write(f"  Recall: {m['recall']:.4f}\n")
            f.write(f"  F1-Score: {m['f1']:.4f}\n")
            f.write(f"  Píxeles Cluster: {res['pixeles_cluster']:,}\n")
            f.write(f"  Píxeles Verdurinipiratini: {res['pixeles_verdurinipiratini']:,}\n")
            f.write(f"  Intersección: {m['intersection']:,}\n")
            f.write(f"  Unión: {m['union']:,}\n")
            f.write("-" * 80 + "\n")
    
    print(f"\n📊 RESUMEN DE MÉTRICAS:")
    print(f"IoU Promedio: {iou_promedio:.4f}")
    print(f"Precision Promedio: {precision_promedio:.4f}")
    print(f"Recall Promedio: {recall_promedio:.4f}")
    print(f"F1-Score Promedio: {f1_promedio:.4f}")

if __name__ == "__main__":
    comparar_clusters_con_verdurinipiratini()
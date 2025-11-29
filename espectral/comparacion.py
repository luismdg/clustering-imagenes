import cv2
import numpy as np
import os

# Configuración inicial
base_dir = os.path.dirname(__file__)

def load_mask(path):
    """Carga máscara desde PNG con transparencia."""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")

    # Requiere canal alpha
    if mask.shape[2] < 4:
        raise ValueError(f"La imagen no tiene canal alpha: {path}")

    alpha = mask[:, :, 3]
    return (alpha > 0).astype(np.uint8)

def compute_iou(m1, m2):
    """Calcula IoU entre dos máscaras binarias."""
    if m1.shape != m2.shape:
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return intersection / union if union > 0 else 0

def compute_dice(m1, m2):
    """Calcula el coeficiente Dice entre dos máscaras binarias."""
    if m1.shape != m2.shape:
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = (m1 & m2).sum()
    return (2.0 * intersection) / (m1.sum() + m2.sum()) if (m1.sum() + m2.sum()) > 0 else 0

# =======================
# LISTA DE PARES A COMPARAR
# =======================
# MODIFICA ESTA LISTA CON LOS PARES QUE QUIERES COMPARAR
# Formato: (ruta_máscara_1, ruta_máscara_2, "Descripción opcional")
pairs = [
    (os.path.join(base_dir, "..", "verdurinipiratini", "ajo.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_simple", "vegetal_01.png"), "Ajo vs Vegetal 01"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "bolalila.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_18.png"), "Bolalila vs Vegetal 18"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "cebollin.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_18.png"), "Cebollin vs Vegetal 18"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chileamarillo.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_04.png"), "Chile Amarillo vs Vegetal 04"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chileazul.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_simple", "vegetal_06.png"), "Chile Azul vs Vegetal 06"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chileazul2.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_21.png"), "Chile Azul 2 vs Vegetal 21"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chileazul3.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_18.png"), "Chile Azul 3 vs Vegetal 18"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chilemorado.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_10.png"), "Chile Morado vs Vegetal 10"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chilenaranja.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_10.png"), "Chile Naranja vs Vegetal 10"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chilerosa.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_14.png"), "Chile Rosa vs Vegetal 14"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chileverdecachi.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_06.png"), "Chile Verde Cachi vs Vegetal 06"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chileverdeclaro.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_03.png"), "Chile Verde Claro vs Vegetal 03"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chileverdelimon.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_16.png"), "Chile Verde Limón vs Vegetal 16"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "chileverdelimon2.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_03.png"), "Chile Verde Limón 2 vs Vegetal 03"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "durazno.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_13.png"), "Durazno vs Vegetal 13"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "manzanarosa.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_07.png"), "Manzana Rosa vs Vegetal 07"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "pimientoamarillo.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_14.png"), "Pimiento Amarillo vs Vegetal 14"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "pimientoazul.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_05.png"), "Pimiento Azul vs Vegetal 05"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "pimientonaranja.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_19.png"), "Pimiento Naranja vs Vegetal 19"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "pimientorojo.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_20.png"), "Pimiento Rojo vs Vegetal 20"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "pimientorojo2.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_01.png"), "Pimiento Rojo 2 vs Vegetal 01"),
    (os.path.join(base_dir, "..", "verdurinipiratini", "pimientoverde.png"), os.path.join(base_dir, "..", "espectral", "mascaras_verduras_corregido", "vegetal_06.png"), "Pimiento Verde vs Vegetal 06s"),
    # Puedes agregar más pares aquí...
]
# =======================
# VALIDACIÓN Y PROCESAMIENTO
# =======================
def validate_path(path):
    """Verifica si la ruta existe."""
    if not os.path.exists(path):
        print(f"ADVERTENCIA: No se encuentra el archivo: {path}")
        return False
    return True

def process_comparisons(pairs_list):
    """Procesa todas las comparaciones en la lista de pares."""
    results = {}
    valid_pairs = []
    
    print("Validando rutas de archivos...")
    
    # Validar que existan todos los archivos
    for i, pair in enumerate(pairs_list):
        if len(pair) == 2:
            path1, path2 = pair
            description = f"Comparación {i+1}"
        elif len(pair) == 3:
            path1, path2, description = pair
        else:
            print(f"Error en el par {i+1}: Formato incorrecto. Usa (path1, path2) o (path1, path2, descripción)")
            continue
        
        if validate_path(path1) and validate_path(path2):
            valid_pairs.append((path1, path2, description))
        else:
            print(f"Saltando par: {description}")
    
    print(f"\nProcesando {len(valid_pairs)} pares válidos...")
    
    # Procesar comparaciones válidas
    for path1, path2, description in valid_pairs:
        try:
            m1 = load_mask(path1)
            m2 = load_mask(path2)

            iou = compute_iou(m1, m2)
            dice = compute_dice(m1, m2)
            
            results[description] = {
                'path1': path1,
                'path2': path2,
                'iou': iou,
                'dice': dice
            }

            #print(f"{description}")
            #print(f"  {os.path.basename(path1)} vs {os.path.basename(path2)}")
            #print(f"  IoU = {iou:.4f}, Dice = {dice:.4f}")
            #print("-" * 50)

        except Exception as e:
            print(f"Error procesando {description}: {e}")
    
    return results

def save_results(results, filename="comparison_results.txt"):
    """Guarda los resultados en un archivo de texto."""
    results_path = os.path.join(base_dir, filename)
    
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("RESULTADOS DE COMPARACIÓN DE MÁSCARAS\n")
        f.write("=" * 50 + "\n\n")
        
        for desc, data in results.items():
            f.write(f"COMPARACIÓN: {desc}\n")
            f.write(f"Archivo 1: {data['path1']}\n")
            f.write(f"Archivo 2: {data['path2']}\n")
            f.write(f"IoU: {data['iou']:.4f}\n")
            f.write(f"Dice: {data['dice']:.4f}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nResultados guardados en: {results_path}")

def print_summary(results):
    """Imprime un resumen de los resultados."""
    if not results:
        print("No hay resultados para mostrar.")
        return
    
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    
    iou_values = [data['iou'] for data in results.values()]
    dice_values = [data['dice'] for data in results.values()]
    
    print(f"Comparaciones realizadas: {len(results)}")
    print(f"IoU promedio: {np.mean(iou_values):.4f}")
    print(f"Dice promedio: {np.mean(dice_values):.4f}")
    print(f"Mejor IoU: {max(iou_values):.4f}")
    print(f"Peor IoU: {min(iou_values):.4f}")
    
    # Imprimir todos los IoU
    print("\nTODAS LAS COMPARACIONES:")
    print("-" * 60)
    for desc, data in results.items():
        print(f"{desc}: IoU = {data['iou']:.4f}, Dice = {data['dice']:.4f}")
    
    # Mostrar las 3 mejores comparaciones
    print("\nTOP 3 MEJORES COMPARACIONES (por IoU):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['iou'], reverse=True)
    for i, (desc, data) in enumerate(sorted_results[:3]):
        print(f"{i+1}. {desc}: IoU = {data['iou']:.4f}")

# =======================
# EJECUCIÓN PRINCIPAL
# =======================
if __name__ == "__main__":
    print("COMPARADOR DE MÁSCARAS - IoU y Dice")
    print("=" * 50)
    
    if not pairs:
        print("No hay pares definidos para comparar.")
        print("\nINSTRUCCIONES:")
        print("1. Modifica la lista 'pairs' en el código")
        print("2. Agrega tus pares en el formato:")
        print("   (ruta_máscara_1, ruta_máscara_2, 'Descripción opcional')")
        print("3. Ejecuta el script nuevamente")
    else:
        results = process_comparisons(pairs)
        
        if results:
            save_results(results)
            print_summary(results)
        else:
            print("No se pudieron procesar las comparaciones. Verifica las rutas de los archivos.")
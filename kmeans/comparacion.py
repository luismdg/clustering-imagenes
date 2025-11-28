import cv2
import numpy as np
import os

# folder for this script (clustering-imagenes/kmeans)
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


# =======================
# LISTA DE PARES A COMPARAR
# =======================

#("", "clustering-imagenes/verdurinipiratini/chileazul.png"),
#("", "clustering-imagenes/verdurinipiratini/chileverdelimon.png"),
#("", "clustering-imagenes/verdurinipiratini/chileverdelimon2.png"),
#("", "clustering-imagenes/verdurinipiratini/manzanarosa.png"),
#("", "clustering-imagenes/verdurinipiratini/pimientoamarillo.png"),
#("", "clustering-imagenes/verdurinipiratini/pimientonaranja.png"),
#(os.path.join(base_dir, "clusters_kmeans", "verdura_15_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chileazul.png"))),
#(os.path.join(base_dir, "clusters_kmeans", "verdura_15_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chileverdelimon.png"))),


pairs = [
    (os.path.join(base_dir, "clusters_kmeans", "verdura_09_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "ajo.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_06_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "bolalila.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_22_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "cebollin.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_10_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chileamarillo.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_15_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chileazul2.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_18_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chileazul3.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_13_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chilemorado.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_17_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chilenaranja.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_12_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chilerosa.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_21_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chileverdecachi.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_14_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "chileverdeclaro.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_23_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "durazno.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_05_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "manzanarosa.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_11_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "pimientoamarillo.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_08_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "pimientoazul.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_07_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "pimientonaranja.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_04_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "pimientorojo.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_16_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "pimientorojo2.png"))),
    (os.path.join(base_dir, "clusters_kmeans", "verdura_21_mask.png"), os.path.normpath(os.path.join(base_dir, "..", "verdurinipiratini", "pimientoverde.png"))),
]

# =======================
# PROCESAR TODAS LAS IMÁGENES
# =======================
results = {}

for m1_path, m2_path in pairs:
    try:
        m1 = load_mask(m1_path)
        m2 = load_mask(m2_path)

        IoU = compute_iou(m1, m2)
        results[(m1_path, m2_path)] = IoU

        print(f"{os.path.basename(m1_path)} vs {os.path.basename(m2_path)} → IoU = {IoU:.4f}")

    except Exception as e:
        print("Error:", e)

# Opcional: guardar resultados dentro de `kmeans`
iou_path = os.path.join(base_dir, "IoU_results.txt")
with open(iou_path, "w") as f:
    for k, v in results.items():
        f.write(f"{k[0]} , {k[1]} , IoU = {v}\n")


'''
import cv2
import numpy as np

mask1_path = os.path.join(base_dir, "clusters_kmeans", "verdura_09_mask.png")
mask2_path = "clustering-imagenes/verdurinipiratini/ajo.png"

# ============ MÁSCARA 1 (PNG con transparencia, objeto negro) ============
mask1 = cv2.imread(mask1_path, cv2.IMREAD_UNCHANGED)

if mask1 is None:
    print("Error cargando máscara 1")

# Extraer ALPHA correctamente
if mask1.shape[2] == 4:
    m1 = mask1[:, :, 3]    # alpha
else:
    raise ValueError("La máscara 1 NO tiene canal alpha")

m1_bin = (m1 > 0).astype(np.uint8)

# ============ MÁSCARA 2 ============
mask2 = cv2.imread(mask2_path, cv2.IMREAD_UNCHANGED)

alpha2 = mask2[:, :, 3]
m2 = (alpha2 > 0).astype(np.uint8)

# Igualar tamaños
if m1_bin.shape != m2.shape:
    m2 = cv2.resize(m2, (m1_bin.shape[1], m1_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

# Debug
cv2.imwrite("debug_m1_fixed.png", m1_bin * 255)
cv2.imwrite("debug_m2_fixed.png", m2 * 255)

# IoU
intersection = (m1_bin & m2).sum()
union = (m1_bin | m2).sum()

IoU = intersection / union if union > 0 else 0

print("IoU =", IoU)
'''
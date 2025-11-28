import cv2
import numpy as np
import json
import os

# ============================
# CONFIG
# ============================
base_dir = os.path.dirname(__file__)
IMAGE_PATH = os.path.normpath(os.path.join(base_dir, "..", "pirataverdu.png"))
OUTPUT_FOLDER = os.path.join(base_dir, "clusters_kmeans")
JSON_PATH = os.path.join(base_dir, "clusters_boundingboxes.json")
FINAL_IMAGE_PATH = os.path.join(base_dir, "verduras_reconstruidas.png")
TARGET_OBJECTS = 24

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================
# 1. LOAD IMAGE
# ============================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("No se pudo cargar la imagen.")

h, w = img.shape[:2]
print(f"Imagen: {w}x{h}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ============================
# 2. K-MEANS SEGMENTATION BY COLOR
# ============================
Z = img_rgb.reshape((-1, 3)).astype(np.float32)

# Try different K values to find 24 objects
for K in [25, 30, 35, 40]:
    print(f"\nProbando K={K} clusters...")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Reshape labels to image
    label_img = labels.reshape((h, w))

    # Find unique labels (excluding background)
    unique_labels = np.unique(label_img)

    # For each cluster, check if it's a valid object
    objects = []

    for label_id in unique_labels:
        # Create mask for this color cluster
        mask = (label_img == label_id).astype(np.uint8) * 255

        # Find contours in this mask
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Each contour could be a separate vegetable
        for c in cnts:
            area = cv2.contourArea(c)

            # Filter by area (adjust if needed)
            min_area = (w * h) * 0.001  # 0.1% of image
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(c)

            # Create individual mask
            obj_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(obj_mask, [c], -1, 255, -1)

            objects.append({
                'area': area,
                'x': x, 'y': y, 'w': bw, 'h': bh,
                'mask': obj_mask,
                'label': int(label_id)
            })

    print(f"   Objetos encontrados: {len(objects)}")

    # Check if we're close to 24
    if len(objects) >= TARGET_OBJECTS:
        print(f"✅ Usando K={K}")
        break

# ============================
# 3. SELECT TOP 24 BY AREA
# ============================
objects = sorted(objects, key=lambda o: o['area'], reverse=True)[:TARGET_OBJECTS]

print(f"\nSeleccionados: {len(objects)} objetos más grandes")

# ============================
# 4. EXPORT INDIVIDUAL MASKS + JSON
# ============================
json_data = []
final_canvas = np.zeros_like(img_rgb)

for i, obj in enumerate(objects):
    x, y, bw, bh = obj['x'], obj['y'], obj['w'], obj['h']
    mask = obj['mask']  # 255 = objeto, 0 = fondo

    # --- Crear PNG con fondo transparente y figura negra ---
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Figura negra
    rgba[:, :, 0] = 0
    rgba[:, :, 1] = 0
    rgba[:, :, 2] = 0

    # El alfa debe ser 255 donde hay objeto
    rgba[:, :, 3] = mask  # 255 objeto, 0 fondo

    mask_file = os.path.join(OUTPUT_FOLDER, f"verdura_{i+1:02d}_mask.png")
    cv2.imwrite(mask_file, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    # Para reconstrucción en canvas final
    final_canvas = np.where(
        mask[:, :, None] == 255,
        img_rgb,
        final_canvas
    )

    json_data.append({
        "id": i + 1,
        "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
        "area": int(obj['area']),
        "cluster": obj['label'],
        "path": mask_file
    })

# ============================
# 5. SAVE OUTPUTS
# ============================
with open(JSON_PATH, "w") as f:
    json.dump(json_data, f, indent=2)

cv2.imwrite(FINAL_IMAGE_PATH, cv2.cvtColor(final_canvas, cv2.COLOR_RGB2BGR))

print(f"\nCOMPLETADO: {len(objects)} verduras")
print(f"Máscaras individuales: {OUTPUT_FOLDER}/")
print(f"JSON: {JSON_PATH}")
print(f"Reconstrucción: {FINAL_IMAGE_PATH}")
print("\nCada PNG tiene el objeto en su posición original con transparencia")
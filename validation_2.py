import os
from ultralytics import YOLO

# Caminho base para salvar os resultados
validation_dir = "F:/yolo/thid_model/validation"

# Cria a pasta 'validation' se não existir
os.makedirs(validation_dir, exist_ok=True)

# Carrega o modelo
model = YOLO("fine_v1.pt")

# Avalia no dataset de validação e salva TUDO em 'validation/val_results'
metrics = model.val(
    data="mask.yaml",
    split="val",  # Usa o dataset de validação
    project=validation_dir,  # Diretório base para salvar os resultados
    name="val_results",       # Nome da pasta dentro de 'validation'
    plots=True,  # Gera gráficos (results.png, confusion_matrix.png, etc.)
    save=True,   # Salva os resultados (imagens, métricas, etc.)
    conf=0.4,    # Threshold de confiança (opcional)
    iou=0.5,     # IOU threshold (opcional)
)

# Extrair valores escalares das métricas
map50 = float(metrics.box.map50[0]) if hasattr(metrics.box.map50, '__len__') else float(metrics.box.map50)
precision = float(metrics.box.p[0]) if hasattr(metrics.box.p, '__len__') else float(metrics.box.p)
recall = float(metrics.box.r[0]) if hasattr(metrics.box.r, '__len__') else float(metrics.box.r)
f1 = float(metrics.box.f1[0]) if hasattr(metrics.box.f1, '__len__') else float(metrics.box.f1)

# Salva as métricas em um arquivo de texto dentro de 'validation'
with open(f"{validation_dir}/val_results/results.txt", "w") as f:
    f.write(f"mAP50: {map50:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")

# Exibe as métricas no console
print(f"mAP50: {map50:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print(f"\nTodos os resultados foram salvos em: {validation_dir}/val_results")
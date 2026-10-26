from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# Caminho do modelo treinado (treino usado no artigo)
MODELO = r"C:\tccyolo26\treinos\plastic_organico_yolo26n-2\weights\best.pt"
IMAGENS = r"C:\tccyolo26\teste_independente\images"

if not Path(MODELO).exists():
    raise FileNotFoundError(f"Modelo não encontrado: {MODELO}")
if not Path(IMAGENS).exists():
    raise FileNotFoundError(f"Pasta de imagens não encontrada: {IMAGENS}")

model = YOLO(MODELO)

results = model.predict(
    source=IMAGENS,
    conf=0.364,        # limiar que maximiza o F1 no artigo (teste também com 0.25)
    imgsz=640,
    save=True,         # salva as imagens com as caixas desenhadas
    project=r"C:\tccyolo26\resultados_teste",
    name="predict",
    exist_ok=True,
)

total = Counter()
sem_deteccao = []

for r in results:
    nome = Path(r.path).name
    classes = [model.names[int(c)] for c in r.boxes.cls]
    total.update(classes)
    if not classes:
        sem_deteccao.append(nome)
    print(f"{nome}: {len(classes)} detecção(ões) {dict(Counter(classes))}")

print("\n" + "=" * 50)
print(f"Imagens analisadas: {len(results)}")
print(f"Detecções por classe: {dict(total)}")
print(f"Imagens sem nenhuma detecção: {len(sem_deteccao)}")
for n in sem_deteccao:
    print(f"  - {n}")
print("Imagens com caixas salvas em: C:\\tccyolo26\\resultados_teste\\predict")
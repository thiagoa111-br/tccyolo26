import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# ---------------- CONFIGURAÇÃO ----------------
ORIGINAL = Path(r"C:\tccyolo26\teste_independente")
CORRIGIDO = Path(r"C:\tccyolo26\teste_independente_corrigido")
SAIDA = Path(r"C:\tccyolo26\resultados_comparacao")

MODELOS = {
    "YOLO26n": r"C:\tccyolo26\treinos\plastic_organico_yolo26n-2\weights\best.pt",
    "YOLOv8n": r"C:\tccyolo26\treinos\plastic_organico_yolov8n\weights\best.pt",
}
# ----------------------------------------------

# 1) Cria uma cópia do teste com os ids 0 <-> 1 trocados (a original não é alterada)
if CORRIGIDO.exists():
    shutil.rmtree(CORRIGIDO)
(CORRIGIDO / "labels").mkdir(parents=True)
shutil.copytree(ORIGINAL / "images", CORRIGIDO / "images")

for txt in (ORIGINAL / "labels").glob("*.txt"):
    novas = []
    for linha in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = linha.split()
        if v:
            v[0] = str(1 - int(float(v[0])))
            novas.append(" ".join(v))
    (CORRIGIDO / "labels" / txt.name).write_text("\n".join(novas) + "\n", encoding="utf-8")

yaml_path = CORRIGIDO / "data.yaml"
yaml_path.write_text(
    "path: C:/tccyolo26/teste_independente_corrigido\n"
    "train: images\nval: images\ntest: images\n\n"
    "nc: 2\nnames:\n  0: plastic\n  1: organico\n",
    encoding="utf-8",
)

# 2) Avalia os dois modelos nas mesmas condições
SAIDA.mkdir(parents=True, exist_ok=True)
linhas = []
for nome, pesos in MODELOS.items():
    print(f"\n===== {nome} =====")
    r = YOLO(pesos).val(
        data=str(yaml_path), imgsz=640, conf=0.001, iou=0.6, batch=1,
        plots=True, project=str(SAIDA), name=nome + "_corrigido", exist_ok=True,
    )
    linha = {
        "modelo": nome,
        "precision": r.box.mp,
        "recall": r.box.mr,
        "mAP50": r.box.map50,
        "mAP50-95": r.box.map,
        "inferencia_ms": r.speed["inference"],
    }
    for i, ap in zip(r.box.ap_class_index, r.box.ap50):
        linha[f"AP50_{r.names[i]}"] = ap
    linhas.append(linha)

df = pd.DataFrame(linhas)
print("\n===== RESULTADO =====")
print(df.set_index("modelo").T)
df.to_csv(SAIDA / "metricas_corrigido.csv", index=False)
print(f"\nGráficos e tabela em: {SAIDA}")
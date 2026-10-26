from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":

    raiz = Path(__file__).resolve().parent
    # Mesmo treino, só trocando a arquitetura
    modelo = YOLO(str(raiz / "yolo11n.pt"))   # baixa automaticamente se não existir

    modelo.train(
        data=str(raiz / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        pretrained=True,
        project=str(raiz / "treinos"),
        name="plastic_organico_yolo11n",   # nome novo
        exist_ok=False,
        plots=True,
        workers=0,
        seed=42,
        deterministic=True
    )

    print("Treinamento concluído!")
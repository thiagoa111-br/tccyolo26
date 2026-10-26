from ultralytics import YOLO

if __name__ == "__main__":

    # Carregar a YOLO26 Nano pré-treinada com COCO
    modelo = YOLO("yolo26n.pt")

    # Iniciar um novo treinamento
    modelo.train(
        data="C:/tccyolo26/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        pretrained=True,
        project="treinos",
        name="plastic_coco",
        exist_ok=False,
        plots=True,
        workers=0
    )

    print("Treinamento concluído!")
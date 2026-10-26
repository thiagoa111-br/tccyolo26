from ultralytics import YOLO

# Carregar a YOLO26
modelo = YOLO("yolo26n.pt")

# Local das fotografias
pasta = r"C:\tccyolo26\original\plastic"

# Detectar os objetos e criar os arquivos TXT
modelo.predict(
    source=pasta,
    conf=0.25,
    save=True,
    save_txt=True,
    save_conf=False,
    project="anotacoes",
    name="teste_plastic"
)

print("Anotação automática concluída!")
import os
import shutil
import random

pasta_imagens = r'C:\tccyolo26\original\plastic'
pasta_labels = r'C:\tccyolo26\original\labels'
pasta_dataset = r'C:\tccyolo26\dataset'

# Criar a estrutura de pastas exigida pelo YOLO
for split in ['train', 'val']:
    os.makedirs(os.path.join(pasta_dataset, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(pasta_dataset, 'labels', split), exist_ok=True)

# Listar todas as imagens
imagens = [f for f in os.listdir(pasta_imagens) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(imagens)

# Divisão 80% treino / 20% validação
split_idx = int(len(imagens) * 0.8)
treino_imgs = imagens[:split_idx]
val_imgs = imagens[split_idx:]

def copiar_ficheiros(lista_imgs, split):
    for img in lista_imgs:
        nome_base = os.path.splitext(img)[0]
        lbl_nome = nome_base + '.txt'
        
        orig_img = os.path.join(pasta_imagens, img)
        orig_lbl = os.path.join(pasta_labels, lbl_nome)
        
        dest_img = os.path.join(pasta_dataset, 'images', split, img)
        dest_lbl = os.path.join(pasta_dataset, 'labels', split, lbl_nome)
        
        shutil.copy(orig_img, dest_img)
        if os.path.exists(orig_lbl):
            shutil.copy(orig_lbl, dest_lbl)

copiar_ficheiros(treino_imgs, 'train')
copiar_ficheiros(val_imgs, 'val')

print("Dataset organizado com sucesso na pasta 'C:\\tccyolo26\\dataset'!")
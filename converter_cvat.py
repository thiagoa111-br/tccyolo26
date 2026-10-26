import xml.etree.ElementTree as ET
import os
import glob

pasta_original = r'C:\tccyolo26\original'
pasta_saida = r'C:\tccyolo26\original\labels'

# Busca automaticamente qualquer ficheiro que comece com 'annotation' na pasta
ficheiros_encontrados = glob.glob(os.path.join(pasta_original, 'annotation*'))

if not ficheiros_encontrados:
    ficheiros_encontrados = glob.glob(os.path.join(pasta_original, '*.xml'))

if not ficheiros_encontrados:
    print("ERRO: Nenhum ficheiro de anotações foi encontrado em C:\\tccyolo26\\original")
    exit()

caminho_xml = ficheiros_encontrados[0]
print(f"Ficheiro de anotações localizado: {caminho_xml}")

os.makedirs(pasta_saida, exist_ok=True)

# Mapeamento da classe definida no CVAT
classes = {'plastic': 0}

tree = ET.parse(caminho_xml)
root = tree.getroot()

total = 0
for image in root.findall('image'):
    img_name = os.path.splitext(os.path.basename(image.attrib['name']))[0]
    width = float(image.attrib['width'])
    height = float(image.attrib['height'])
    
    txt_filename = os.path.join(pasta_saida, f"{img_name}.txt")
    
    with open(txt_filename, 'w') as f:
        for box in image.findall('box'):
            label = box.attrib['label']
            if label in classes:
                cls_id = classes[label]
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                
                # Conversão para coordenadas normalizadas do YOLO
                x_center = ((xtl + xbr) / 2) / width
                y_center = ((ytl + ybr) / 2) / height
                w = (xbr - xtl) / width
                h = (ybr - ytl) / height
                
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    total += 1

print(f"Sucesso! Foram gerados {total} ficheiros .txt na pasta: {pasta_saida}")
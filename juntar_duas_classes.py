"""Combina os dados existentes sem modificar imagens ou anotações de origem."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'dataset'
DEST = ROOT / 'dataset_duas_classes'


def main():
    if DEST.exists():
        raise SystemExit(f'O destino já existe: {DEST}. Nenhum arquivo foi alterado.')
    records = []
    errors = []
    sources = []
    for split in ('train', 'val'):
        for image in sorted((SOURCE / 'images' / split).glob('*.jpg')):
            sources.append((image, SOURCE / 'labels' / split / (image.stem+'.txt'), 'plastic', split))
    for image in sorted((SOURCE / 'biological').glob('*.jpg')):
        sources.append((image, SOURCE / 'labels/train' / (image.stem+'.txt'), 'organic', None))
    for index, (image, label, origin, split) in enumerate(sources):
        if not label.exists():
            errors.append(f'Label ausente: {label}')
            continue
        try:
            with Image.open(image) as im:
                im.load()
                width, height = im.size
            if min(width, height) <= 0:
                raise ValueError('Dimensões inválidas')
            mapped = []
            counts = Counter()
            for number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'Linha {number}: esperado classe x y largura altura')
                class_id = int(fields[0])
                coords = list(map(float, fields[1:]))
                x, y, w, h = coords
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or min(w,h) <= 0:
                    raise ValueError(f'Linha {number}: coordenadas inválidas')
                if x-w/2 < -1e-5 or y-h/2 < -1e-5 or x+w/2 > 1+1e-5 or y+h/2 > 1+1e-5:
                    raise ValueError(f'Linha {number}: caixa fora da imagem')
                mapping = {0:0} if origin == 'plastic' else {0:1, 1:0}
                if class_id not in mapping:
                    raise ValueError(f'Linha {number}: classe inesperada {class_id}')
                new_id = mapping[class_id]
                mapped.append(' '.join([str(new_id), *fields[1:]]))
                counts[new_id] += 1
            records.append(dict(image=str(image), label=str(label), origin=origin,
                                previous_split=split, split=split, counts=dict(counts),
                                sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                                content='\n'.join(mapped)+ ('\n' if mapped else '')))
        except Exception as exc:
            errors.append(f'{image.name}: {exc}')
        if (index+1) % 500 == 0:
            print(f'Validados {index+1}/{len(sources)} pares', flush=True)
    if errors:
        (ROOT/'erros_dataset_duas_classes.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2),encoding='utf-8')
        raise SystemExit(f'{len(errors)} erros. Consulte erros_dataset_duas_classes.json. Destino não criado.')
    if not records or not any(r['origin']=='organic' for r in records):
        raise SystemExit('Não foram encontrados dados de ambas as origens.')
    if len({Path(r['image']).name for r in records}) != len(records):
        raise SystemExit('Nomes de imagem repetidos; união cancelada para evitar sobrescrita.')
    groups = defaultdict(list)
    for r in records:
        groups[r['sha256']].append(r)
    organic_groups=[]
    for digest, group in groups.items():
        existing={r['previous_split'] for r in group if r['previous_split']}
        if existing:
            # Em caso de cópia exata em ambas as divisões, manter o grupo na validação.
            split='val' if 'val' in existing else 'train'
            for r in group:r['split']=split
        else:
            organic_groups.append(digest)
    random.Random(42).shuffle(organic_groups)
    cut=int(len(organic_groups)*.8)
    for i,digest in enumerate(organic_groups):
        for r in groups[digest]:r['split']='train' if i<cut else 'val'
    assert not ({r['sha256'] for r in records if r['split']=='train'} & {r['sha256'] for r in records if r['split']=='val'})
    summary={}
    for split in ('train','val'):
        group=[r for r in records if r['split']==split]
        instances=Counter()
        for r in group:instances.update(r['counts'])
        if not instances[0] or not instances[1]:
            raise SystemExit(f'Uma classe ficou sem instâncias em {split}')
        summary[split]={'images':len(group),'source_images':dict(Counter(r['origin'] for r in group)),
                        'instances':{'plastic':instances[0],'organico':instances[1]},
                        'empty_labels':sum(not r['content'] for r in group)}
    for split in ('train','val'):
        for kind in ('images','labels'):(DEST/kind/split).mkdir(parents=True)
    for r in records:
        image=Path(r['image'])
        shutil.copy2(image, DEST/'images'/r['split']/image.name)
        (DEST/'labels'/r['split']/(image.stem+'.txt')).write_text(r.pop('content'),encoding='utf-8')
    yaml=f'path: {DEST.as_posix()}\ntrain: images/train\nval: images/val\nnc: 2\nnames:\n  0: plastic\n  1: organico\n'
    (DEST/'data.yaml').write_text(yaml,encoding='utf-8')
    manifest={'names':{0:'plastic',1:'organico'},'organic_split_seed':42,'summary':summary,
              'identical_image_groups':sum(len(g)>1 for g in groups.values()),
              'moved_existing_images':sum(r['previous_split'] is not None and r['previous_split']!=r['split'] for r in records),
              'records':records}
    (DEST/'manifesto.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='records'},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()

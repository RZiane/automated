import os
from conllu import parse

corpus = []

def load_conllu_file(file_path, output_file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        
        sentences = parse(data)

        for sent in sentences:
            print(sent.metadata)
            if sent.metadata['newdoc_id'] == 'Roland_1100_verse':
                corpus.append(sent)
    
    # Écrire les résultats dans le fichier de sortie
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        # Utiliser writelines pour écrire chaque ligne séparément
        outfile.writelines([sentence.serialize() + "" for sentence in corpus])

file_path = '/home/ziane212/projects/UD_Old_French-PROFITEROLE-master/fro_profiterole-ud-train.conllu'
output_file_path = '/home/ziane212/projects/UD_Old_French-PROFITEROLE-master/fro_profiterole-ud-train_rolland.conllu'
load_conllu_file(file_path, output_file_path)
import os
import conllu
import shutil

def concat_conllu_files(folder_path, output_file):
    # Vérifie que le dossier existe
    if not os.path.isdir(folder_path):
        print("Le chemin spécifié n'est pas un dossier valide.")
        return
    
    # Initialise une liste pour stocker les phrases concaténées
    concatenated_sentences = []
    
    # Parcours les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        if filename.endswith(".conllu"):
            file_path = os.path.join(folder_path, filename)
            # Ouvre chaque fichier et concatène les phrases
            with open(file_path, "r", encoding="utf-8") as file:
                data = file.read()
                sentences = conllu.parse(data)
                for i in sentences:
                    # i.metadata['sent_id'] = filename+'_'+i.metadata['sent_id']
                    i.metadata['sent_id'] = i.metadata['sent_id']
                concatenated_sentences.extend(sentences)
    
    # Écrit les phrases concaténées dans un fichier de sortie
    with open(output_file, "w", encoding="utf-8") as output:
        output.writelines([sentence.serialize() for sentence in concatenated_sentences])

concat_conllu_files("/home/or-llsh-156-l01/projets/CRISCO/Rome_eval/dump_Rome_II_parsing_Final/validated_data_eval",
                    "/home/or-llsh-156-l01/projets/CRISCO/Rome_eval/dump_Rome_II_parsing_Final/validated_data_eval/all.conllu")
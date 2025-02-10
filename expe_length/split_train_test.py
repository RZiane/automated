import os
import conllu
import random
import math
import shutil

def split_conllu(input_file, output_train_file, output_test_file, proportion_test):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    sentences = list(conllu.parse(data))
    
    num_test_sentences = math.ceil(len(sentences) * proportion_test)
    # num_test_sentences = 150

    # Mélanger les phrases de manière aléatoire
    random.shuffle(sentences)

    # Séparer les phrases en ensembles d'entraînement et de test
    test_set = sentences[:num_test_sentences]
    train_set = sentences[num_test_sentences:]

    # Écriture des phrases dans les fichiers de sortie
    with open(output_train_file, 'w', encoding='utf-8') as train_file:
        train_file.writelines([sentence.serialize() + "" for sentence in train_set])

    with open(output_test_file, 'w', encoding='utf-8') as test_file:
        # test_file.write(conllu.serialize(test_set))
        test_file.writelines([sentence.serialize() + "" for sentence in test_set])

def process_directory(input_directory, output_directory_train, output_directory_test, proportion_test):
    # Créer un sous-dossier pour les fichiers de sortie s'il n'existe pas
    if not os.path.exists(output_directory_train):
        os.makedirs(output_directory_train)
    if not os.path.exists(output_directory_test):
        os.makedirs(output_directory_test)

    # Parcourir tous les fichiers du répertoire d'entrée
    for filename in os.listdir(input_directory):
        if filename.endswith(".conllu"):
            input_filepath = os.path.join(input_directory, filename)
            output_train_filepath = os.path.join(output_directory_train, f"{os.path.splitext(filename)[0]}_train.conllu")
            output_test_filepath = os.path.join(output_directory_test, f"{os.path.splitext(filename)[0]}_test.conllu")

            # Appliquer la segmentation pour chaque fichier du répertoire
            split_conllu(input_filepath, output_train_filepath, output_test_filepath, proportion_test)

def regrouper_fichiers_par_groupe(dossiers_sources, dossier_destination):
    # Créer le dossier de destination s'il n'existe pas
    if not os.path.exists(dossier_destination):
        os.makedirs(dossier_destination)
    
    # Parcourir les dossiers sources
    for dossier_source in dossiers_sources:
        # Vérifier si le chemin existe
        if os.path.exists(dossier_source):
            print(f"Traitement du dossier : {dossier_source}")
            
            # Parcourir les fichiers dans le dossier source
            for fichier in os.listdir(dossier_source):
                chemin_fichier = os.path.join(dossier_source, fichier)
                
                # Vérifier si c'est un fichier .conllu
                if os.path.isfile(chemin_fichier) and fichier.endswith(".conllu"):
                    # Parcourir tous les groupes possibles de 0 à 10
                    for groupe in range(11):
                        # Vérifier si le numéro de groupe est dans le nom du fichier
                        if f"_{groupe}_" in fichier:
                            # Créer un dossier pour ce groupe s'il n'existe pas
                            dossier_groupe = os.path.join(dossier_destination, f"group{groupe}")
                            if not os.path.exists(dossier_groupe):
                                os.makedirs(dossier_groupe)
                            
                            # Copier le fichier dans le dossier du groupe
                            shutil.copy(chemin_fichier, dossier_groupe)
                            print(f"Copié {fichier} dans {dossier_groupe}")
                            # Si le numéro de groupe est trouvé, pas besoin de vérifier les autres groupes
                            break
        else:
            print(f"Dossier source inexistant : {dossier_source}")

# Utilisation de la fonction
# dossiers_sources = [ "/home/ziane212/projects/AUTOMATED/bootstrapping_SRCMF/data/frm/size_split_commyn/train",
#                     "/home/ziane212/projects/AUTOMATED/bootstrapping_SRCMF/data/frm/size_split_grchronj2c5/train",
#                     "/home/ziane212/projects/AUTOMATED/bootstrapping_SRCMF/data/frm/size_split_Jehpar/train"]
# dossier_destination = "/home/ziane212/projects/AUTOMATED/bootstrapping_SRCMF/data/frm/split_group"
# regrouper_fichiers_par_groupe(dossiers_sources, dossier_destination)

# Utilisation du script pour un répertoire
input_directory_path = "/home/or-llsh-156-l01/projets/CRISCO/Gascon/train/split_size"
output_directory_train = "/home/or-llsh-156-l01/projets/CRISCO/Gascon/trash"
output_directory_test = "//home/or-llsh-156-l01/projets/CRISCO/Gascon/train/split_size/gr3"
proportion_test = 0.40 # 20% du nombre total de phrases sera extrait pour chaque fichier

process_directory(input_directory_path, output_directory_train, output_directory_test, proportion_test)

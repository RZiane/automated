#!/bin/bash

# Chemin du dossier des modèles
models_path="/home/ziane212/projects/models"

# Chemin du dossier des fichiers d'entrée
input_files_path="/home/ziane212/projects/AUTOMATED/bootstrapping_SRCMF/data/test_eval"

csv_output="/home/ziane212/projects/AUTOMATED/conll18_ud_eval.csv"
echo "Input_file,Model,LAS,UAS" > "$csv_output"

# Parcours des dossiers dans le dossier des modèles
for model_dir in "$models_path"/*/; do
    # Vérifier si le chemin correspond à un dossier
    if [ -d "$model_dir" ]; then
        # Extraire le nom du modèle à partir du nom du dossier
        model_name=$(basename -- "$model_dir")

        # Créer un sous-dossier de sortie pour le modèle
        output_dir="$input_files_path/out/${model_name}_out"
        mkdir -p "$output_dir"

        # Parcours des fichiers dans le dossier des fichiers d'entrée
        for input_file in "$input_files_path"/*; do
            # Vérifier si le chemin correspond à un fichier
            if [ -f "$input_file" ]; then
                # Extraire le nom du fichier et son extension
                filename=$(basename -- "$input_file")
                extension="${filename##*.}"
                filename_no_ext="${filename%.*}"

                # Chemin du fichier de sortie
                output_file="$output_dir/${filename_no_ext}_out.$extension"

                # Exécuter la commande en utilisant le chemin du modèle, du fichier d'entrée et du fichier de sortie appropriés
                hopsparser parse "$model_dir" "$input_file" "$output_file" 
                #"--device=cuda"

                # Exécuter le script d'évaluation avec les fichiers d'entrée et de sortie
                evaluation_output=$(/home/ziane212/anaconda3/bin/python /home/ziane212/projects/AUTOMATED/exe_hops_serie.sh "$input_file" "$output_file")

                # Extraire les scores de l'évaluation
                LAS_score=$(echo "$evaluation_output" | awk 'NR==1{print $4}')
                UAS_score=$(echo "$evaluation_output" | awk 'NR==2{print $4}')

                # Ajouter les données dans le fichier CSV (une seule ligne pour chaque modèle et fichier d'entrée)
                echo "$input_file,$model_dir,$LAS_score,$UAS_score" >> "$csv_output"
            fi
        done
    fi
done

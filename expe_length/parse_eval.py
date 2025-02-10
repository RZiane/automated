import os
import csv
import conllu
import conll18_ud_eval as eval
from sklearn.metrics import classification_report
import pandas as pd
from collections import defaultdict
from conllu import parse_incr
# import torch
# import gc
import argparse

# if torch.cuda.is_available():
#     print("Good, everything is working fine (so far) :).")
# else:
#     print("GPU is not available !")

# torch.cuda.empty_cache()
# gc.collect()

# Fonction pour lire un fichier CoNLL-U et extraire les dépendances
def read_conllu(file_path):
    dependencies = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            for token in tokenlist:
                if 'head' in token and 'deprel' in token:
                    head = token['head']
                    deprel = token['deprel']
                    dependencies.append((head, deprel))
    return dependencies

# Fonction pour lire un fichier CoNLL-U
def read_conllu_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return conllu.parse(f.read())

# Fonction pour extraire les tags (déprélations dans ce cas)
def extract_tags(conllu_data):
    return [token['deprel'] for sentence in conllu_data for token in sentence]

# Fonction pour calculer les métriques par déprel
def calculate_metrics_by_deprel(gold, predicted):
    correct_by_deprel = defaultdict(int)
    predicted_by_deprel = defaultdict(int)
    gold_by_deprel = defaultdict(int)

    for (gold_head, gold_deprel), (pred_head, pred_deprel) in zip(gold, predicted):
        predicted_by_deprel[pred_deprel] += 1
        gold_by_deprel[gold_deprel] += 1
        if gold_head == pred_head and gold_deprel == pred_deprel:
            correct_by_deprel[gold_deprel] += 1

    metrics_by_deprel = {}

    for deprel in gold_by_deprel:
        precision = (correct_by_deprel[deprel] / predicted_by_deprel[deprel] * 100) if predicted_by_deprel[deprel] > 0 else 0.0
        recall = (correct_by_deprel[deprel] / gold_by_deprel[deprel] * 100) if gold_by_deprel[deprel] > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics_by_deprel[deprel] = f1_score

    return metrics_by_deprel

# Fonction pour évaluer les prédictions POS (F1 par classe)
def evaluate_class(reference_file, prediction_file):
    reference_data = read_conllu_file(reference_file)
    reference_pos_tags = extract_tags(reference_data)

    prediction_data = read_conllu_file(prediction_file)
    prediction_pos_tags = extract_tags(prediction_data)

    if len(reference_pos_tags) != len(prediction_pos_tags):
        raise ValueError(f"Mismatch in number of tags between reference and prediction {prediction_file}")

    report = classification_report(reference_pos_tags, prediction_pos_tags, output_dict=True)
    return report

# Fonction principale
def main(args):
    models_path = args.models_path
    input_files_path = args.input_files_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    csv_output = os.path.join(output_path, "eval_scores.csv")
    combined_csv_output = os.path.join(output_path, "combined_classification_report.csv")
    las_csv_output = os.path.join(output_path, "las_by_class_report.csv")

    combined_df = pd.DataFrame()
    las_combined_df = pd.DataFrame()

    with open(csv_output, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Input_file", "Model", "LAS", "UAS", "DEPREL", "UPOS"])

    if args.mode == 'evaluate':
        gold_dir = args.gold_dir
        system_dir = args.system_dir

        for gold_file in [os.path.join(gold_dir, f) for f in os.listdir(gold_dir) if os.path.isfile(os.path.join(gold_dir, f))]:
            filename = os.path.basename(gold_file)
            system_file = os.path.join(system_dir, filename)

            if os.path.isfile(system_file):
                print(f"Evaluating: File={filename}")
                gold = eval.load_conllu_file(gold_file)
                pred = eval.load_conllu_file(system_file)
                evaluation = eval.evaluate(gold, pred)

                UPOS_score = str(evaluation['UPOS'].f1 * 100).replace('.', ',')
                DEPREL_score = str(evaluation['DEPREL'].f1 * 100).replace('.', ',')
                LAS_score = str(evaluation['LAS'].f1 * 100).replace('.', ',')
                UAS_score = str(evaluation['UAS'].f1 * 100).replace('.', ',')

                with open(csv_output, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file, delimiter=';')
                    writer.writerow([filename, "N/A", LAS_score, UAS_score, DEPREL_score, UPOS_score])

                report = evaluate_class(gold_file, system_file)
                class_f1_scores = {class_name: metrics['f1-score'] * 100 for class_name, metrics in report.items() if class_name != 'accuracy'}
                df_report = pd.DataFrame(class_f1_scores, index=[filename])
                combined_df = pd.concat([combined_df, df_report], axis=0)

                gold_dependencies = read_conllu(gold_file)
                predicted_dependencies = read_conllu(system_file)
                las_by_deprel = calculate_metrics_by_deprel(gold_dependencies, predicted_dependencies)

                df_las = pd.DataFrame(las_by_deprel, index=[filename])
                las_combined_df = pd.concat([las_combined_df, df_las], axis=0)

    elif args.mode in ['parse', 'both']:
        for model_dir in [os.path.join(models_path, d) for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]:
            model_name = os.path.basename(model_dir)

            output_dir = os.path.join(output_path, f"{model_name}_out")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            for input_file in [os.path.join(input_files_path, f) for f in os.listdir(input_files_path) if os.path.isfile(os.path.join(input_files_path, f))]:
                filename = os.path.basename(input_file)
                output_file = os.path.join(output_dir, filename)

                try:
                    print(f"Parsing: Model={model_name}, File={filename}")
                    os.system(f"python /home/ziane212/projects/BertForDeprel/BertForDeprel/run.py predict --gpu_ids -2 --model_path {model_dir} --inpath {input_file} --outpath {output_dir} --overwrite --keep_lemmas ALL")

                    if args.mode == 'both':
                        print(f"Evaluating: Model={model_name}, File={filename}")
                        gold = eval.load_conllu_file(input_file)
                        pred = eval.load_conllu_file(output_file)
                        evaluation = eval.evaluate(gold, pred)

                        UPOS_score = str(evaluation['UPOS'].f1 * 100).replace('.', ',')
                        DEPREL_score = str(evaluation['DEPREL'].f1 * 100).replace('.', ',')
                        LAS_score = str(evaluation['LAS'].f1 * 100).replace('.', ',')
                        UAS_score = str(evaluation['UAS'].f1 * 100).replace('.', ',')

                        with open(csv_output, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file, delimiter=';')
                            writer.writerow([filename, model_name, LAS_score, UAS_score, DEPREL_score, UPOS_score])

                        report = evaluate_class(input_file, output_file)
                        class_f1_scores = {class_name: metrics['f1-score'] * 100 for class_name, metrics in report.items() if class_name != 'accuracy'}
                        df_report = pd.DataFrame(class_f1_scores, index=[filename])
                        combined_df = pd.concat([combined_df, df_report], axis=0)

                        gold_dependencies = read_conllu(input_file)
                        predicted_dependencies = read_conllu(output_file)
                        las_by_deprel = calculate_metrics_by_deprel(gold_dependencies, predicted_dependencies)

                        df_las = pd.DataFrame(las_by_deprel, index=[filename])
                        las_combined_df = pd.concat([las_combined_df, df_las], axis=0)

                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {filename} avec le modèle {model_name}: {e}")

    combined_df = combined_df.reset_index().rename(columns={'index': 'Input_file'})
    combined_df.to_csv(combined_csv_output, index=False, sep=";")

    las_combined_df = las_combined_df.reset_index().rename(columns={'index': 'Input_file'})
    las_combined_df.to_csv(las_csv_output, index=False, sep=";")

    print("Traitement terminé et résultats enregistrés.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de parsing et d'évaluation.")
    parser.add_argument("--mode", type=str, choices=['parse', 'evaluate', 'both'], required=True, help="Mode d'exécution: parse, evaluate, ou both.")
    parser.add_argument("--models_path", type=str, help="Chemin des modèles (requis pour parse et both).")
    parser.add_argument("--input_files_path", type=str, help="Chemin des fichiers d'entrée (requis pour parse et both).")
    parser.add_argument("--gold_dir", type=str, help="Chemin des fichiers gold (requis pour evaluate).")
    parser.add_argument("--system_dir", type=str, help="Chemin des fichiers système (requis pour evaluate).")
    parser.add_argument("--output_path", type=str, required=True, help="Chemin du répertoire de sortie.")

    args = parser.parse_args()
    main(args)

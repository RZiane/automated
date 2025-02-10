import os
import csv
import conllu
import conll18_ud_eval as eval
from sklearn.metrics import classification_report
import pandas as pd
from collections import defaultdict
from conllu import parse_incr
import argparse

# ------------------------------------------------------------------------------
# 1) Lecture des phrases (avec métadonnées) depuis un fichier CoNLL-U
# 2) Calcul du LAS (F1) uniquement sur les tokens dont l'ID est listé 
#    dans la métadonnée 'eval' des phrases gold
# ------------------------------------------------------------------------------
def read_conllu_sentences(file_path):
    """
    Lit un fichier CoNLL-U et renvoie la liste des 'Sentence' 
    (objets conllu) avec leurs métadonnées et tokens.
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            sentences.append(tokenlist)
    return sentences

def calculate_las_for_eval_tokens(gold_sentences, pred_sentences):
    """
    Calcule un LAS (F1) basé UNIQUEMENT sur les tokens dont les IDs 
    sont indiqués dans la métadonnée 'eval' de CHAQUE phrase gold.

    Format attendu pour la métadonnée dans la phrase gold :
      # eval = 1;3
    signifie : "n'évaluer que les tokens d'ID 1 et 3 dans cette phrase."

    Hypothèses / conditions :
    - gold_sentences et pred_sentences ont même nombre de phrases, alignées.
    - Les IDs de tokens correspondent (on ignore les multiword tokens, ex 1-2).
    """
    nb_arcs_gold = 0
    nb_arcs_pred = 0
    nb_arcs_corrects = 0

    for gold_sentence, pred_sentence in zip(gold_sentences, pred_sentences):
        # Récupération de la métadonnée eval (ex: "1;3")
        eval_ids = set()
        if 'eval' in gold_sentence.metadata:
            eval_str = gold_sentence.metadata['eval']
            # Ex: "1;3" -> [1, 3]
            eval_ids = {int(x) for x in eval_str.split(';') if x.strip()}

        # On mappe les tokens prédits par leur ID
        pred_map = {}
        for token in pred_sentence:
            # On ignore les éventuels multiword tokens du type "1-2"
            if isinstance(token['id'], int):
                pred_map[token['id']] = (token['head'], token['deprel'])

        # Parcours des tokens gold
        for gold_token in gold_sentence:
            if isinstance(gold_token['id'], int) and gold_token['id'] in eval_ids:
                # On cible seulement les tokens dont l'ID est dans eval
                gold_head = gold_token['head']
                gold_deprel = gold_token['deprel']
                nb_arcs_gold += 1

                # Vérif dans la prédiction
                if gold_token['id'] in pred_map:
                    pred_head, pred_deprel = pred_map[gold_token['id']]
                    nb_arcs_pred += 1

                    # Correct si (head + deprel) match exact
                    if gold_head == pred_head and gold_deprel == pred_deprel:
                        nb_arcs_corrects += 1

    # Calcul F1
    precision = (nb_arcs_corrects / nb_arcs_pred * 100) if nb_arcs_pred > 0 else 0.0
    recall = (nb_arcs_corrects / nb_arcs_gold * 100) if nb_arcs_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1

# ------------------------------------------------------------------------------
# Fonctions existantes (lecture basique pour dépendances, etc.)
# ------------------------------------------------------------------------------
def read_conllu(file_path):
    """
    Lit un fichier CoNLL-U et retourne UNE liste de tuples (head, deprel) 
    pour l'ensemble du fichier (toutes les phrases concaténées).
    """
    dependencies = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            for token in tokenlist:
                if 'head' in token and 'deprel' in token:
                    head = token['head']
                    deprel = token['deprel']
                    dependencies.append((head, deprel))
    return dependencies

def read_conllu_file(filepath):
    """
    Lit un fichier CoNLL-U et retourne sa représentation parsée (liste de phrases),
    chaque phrase étant un objet conllu (tokens + métadonnées).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return conllu.parse(f.read())

def extract_tags(conllu_data):
    """
    Extrait toutes les 'deprel' d'un objet CoNLL-U parsé (plusieurs phrases).
    """
    return [token['deprel'] for sentence in conllu_data for token in sentence]

def calculate_LAS_by_deprel(gold, predicted):
    """
    Calcule le F1-score par déprel (Labeled Attachment) 
    en se basant sur la correspondance (head, relation).
    """
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

def evaluate_class(reference_file, prediction_file):
    """
    Retourne le rapport (dict) du classification_report (F1 par classe),
    ici pour les dépréls.
    """
    reference_data = read_conllu_file(reference_file)
    reference_pos_tags = extract_tags(reference_data)

    prediction_data = read_conllu_file(prediction_file)
    prediction_pos_tags = extract_tags(prediction_data)

    if len(reference_pos_tags) != len(prediction_pos_tags):
        raise ValueError(f"Mismatch in number of tags between reference ({reference_file}) and prediction ({prediction_file}).")

    report = classification_report(reference_pos_tags, prediction_pos_tags, output_dict=True)
    return report

# ------------------------------------------------------------------------------
# 3) Main (parsing + evaluation)
# ------------------------------------------------------------------------------
def main(args):
    # Création du répertoire de sortie si besoin
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    csv_output = os.path.join(args.output_path, "eval_scores.csv")
    combined_csv_output = os.path.join(args.output_path, "combined_classification_report.csv")
    las_csv_output = os.path.join(args.output_path, "las_by_class_report.csv")

    combined_df = pd.DataFrame()
    las_combined_df = pd.DataFrame()

    # On crée le CSV principal (ou on l'écrase s'il existe)
    with open(csv_output, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Input_file", "Model", "LAS", "UAS", "DEPREL", "UPOS", "LAS_Selected", "LAS_EvalTokens"])

    # --------------------------------------------------------------------------
    # MODE = evaluate
    # --------------------------------------------------------------------------
    if args.mode == 'evaluate':
        gold_dir = args.gold_dir
        system_dir = args.system_dir

        for gold_file in [
            os.path.join(gold_dir, f) for f in os.listdir(gold_dir) 
            if os.path.isfile(os.path.join(gold_dir, f))
        ]:
            filename = os.path.basename(gold_file)
            system_file = os.path.join(system_dir, filename)

            if not os.path.isfile(system_file):
                continue

            print(f"Evaluating: File={filename}")
            # 1) Évaluation globale avec conll18_ud_eval
            gold = eval.load_conllu_file(gold_file)
            pred = eval.load_conllu_file(system_file)
            evaluation = eval.evaluate(gold, pred)

            UPOS_score = str(evaluation['UPOS'].f1 * 100).replace('.', ',')
            DEPREL_score = str(evaluation['DEPREL'].f1 * 100).replace('.', ',')
            LAS_score = str(evaluation['LAS'].f1 * 100).replace('.', ',')
            UAS_score = str(evaluation['UAS'].f1 * 100).replace('.', ',')

            # 2) Lecture pour LAS par deprel
            gold_dependencies = read_conllu(gold_file)
            predicted_dependencies = read_conllu(system_file)

            # 3) Calcul du LAS sur la métadonnée "eval"
            #    -> On lit toutes les phrases gold + system
            gold_sentences = read_conllu_sentences(gold_file)
            pred_sentences = read_conllu_sentences(system_file)
            las_eval_tokens = calculate_las_for_eval_tokens(gold_sentences, pred_sentences)
            las_eval_tokens_str = str(las_eval_tokens).replace('.', ',')

            # 4) On écrit dans le CSV principal
            with open(csv_output, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow([
                    filename, 
                    "N/A", 
                    LAS_score, 
                    UAS_score, 
                    DEPREL_score, 
                    UPOS_score,
                    las_eval_tokens_str
                ])

            # 5) Rapport de classification par déprel (F1)
            report = evaluate_class(gold_file, system_file)
            class_f1_scores = {
                class_name: metrics['f1-score'] * 100
                for class_name, metrics in report.items()
                if class_name != 'accuracy'
            }
            df_report = pd.DataFrame(class_f1_scores, index=[filename])
            combined_df = pd.concat([combined_df, df_report], axis=0)

            # 6) LAS par déprel
            las_by_deprel = calculate_LAS_by_deprel(gold_dependencies, predicted_dependencies)
            df_las = pd.DataFrame(las_by_deprel, index=[filename])
            las_combined_df = pd.concat([las_combined_df, df_las], axis=0)

    # --------------------------------------------------------------------------
    # MODE = parse ou both
    # --------------------------------------------------------------------------
    elif args.mode in ['parse', 'both']:
        # On va parser chaque fichier input_files_path avec chacun des modèles
        models_path = args.models_path
        input_files_path = args.input_files_path

        for model_dir in [
            os.path.join(models_path, d) for d in os.listdir(models_path) 
            if os.path.isdir(os.path.join(models_path, d))
        ]:
            model_name = os.path.basename(model_dir)
            output_dir = os.path.join(args.output_path, f"{model_name}_out")
            os.makedirs(output_dir, exist_ok=True)

            for input_file in [
                os.path.join(input_files_path, f) for f in os.listdir(input_files_path) 
                if os.path.isfile(os.path.join(input_files_path, f))
            ]:
                filename = os.path.basename(input_file)
                output_file = os.path.join(output_dir, filename)

                try:
                    # 1) Parsing
                    print(f"Parsing: Model={model_name}, File={filename}")
                    cmd = (
                        f"python /home/ziane212/projects/BertForDeprel/BertForDeprel/run.py "
                        f"predict --gpu_ids -2 --model_path {model_dir} "
                        f"--inpath {input_file} --outpath {output_dir} "
                        f"--overwrite --keep_lemmas ALL"
                    )
                    os.system(cmd)

                    # 2) Si mode=both, on évalue direct
                    if args.mode == 'both':
                        print(f"Evaluating: Model={model_name}, File={filename}")
                        gold = eval.load_conllu_file(input_file)
                        pred = eval.load_conllu_file(output_file)
                        evaluation = eval.evaluate(gold, pred)

                        UPOS_score = str(evaluation['UPOS'].f1 * 100).replace('.', ',')
                        DEPREL_score = str(evaluation['DEPREL'].f1 * 100).replace('.', ',')
                        LAS_score = str(evaluation['LAS'].f1 * 100).replace('.', ',')
                        UAS_score = str(evaluation['UAS'].f1 * 100).replace('.', ',')

                        # 3) lecture pour LAS par déprels
                        gold_dependencies = read_conllu(input_file)
                        predicted_dependencies = read_conllu(output_file)

                        # 4) LAS basé sur la métadonnée 'eval'
                        gold_sentences = read_conllu_sentences(input_file)
                        pred_sentences = read_conllu_sentences(output_file)
                        las_eval_tokens = calculate_las_for_eval_tokens(gold_sentences, pred_sentences)
                        las_eval_tokens_str = str(las_eval_tokens).replace('.', ',')

                        # 5) Enregistrement des scores dans le CSV principal
                        with open(csv_output, mode='a', newline='', encoding='utf-8') as fcsv:
                            writer = csv.writer(fcsv, delimiter=';')
                            writer.writerow([
                                filename, 
                                model_name, 
                                LAS_score, 
                                UAS_score, 
                                DEPREL_score, 
                                UPOS_score,
                                las_eval_tokens_str
                            ])

                        # 6) Rapport par classe
                        report = evaluate_class(input_file, output_file)
                        class_f1_scores = {
                            class_name: metrics['f1-score'] * 100 
                            for class_name, metrics in report.items() 
                            if class_name != 'accuracy'
                        }
                        df_report = pd.DataFrame(class_f1_scores, index=[filename])
                        combined_df = pd.concat([combined_df, df_report], axis=0)

                        # 7) LAS par déprel
                        las_by_deprel = calculate_LAS_by_deprel(gold_dependencies, predicted_dependencies)
                        df_las = pd.DataFrame(las_by_deprel, index=[filename])
                        las_combined_df = pd.concat([las_combined_df, df_las], axis=0)

                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {filename} avec le modèle {model_name}: {e}")

    # --------------------------------------------------------------------------
    # Sauvegarde des rapports globaux (classification + LAS par déprel)
    # --------------------------------------------------------------------------
    combined_df = combined_df.reset_index().rename(columns={'index': 'Input_file'})
    combined_df.to_csv(combined_csv_output, index=False, sep=";")

    las_combined_df = las_combined_df.reset_index().rename(columns={'index': 'Input_file'})
    las_combined_df.to_csv(las_csv_output, index=False, sep=";")

    print("Traitement terminé et résultats enregistrés.")

# ------------------------------------------------------------------------------
# 4) Point d'entrée
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de parsing et d'évaluation.")
    parser.add_argument("--mode", type=str, choices=['parse', 'evaluate', 'both'], required=True,
                        help="Mode d'exécution: parse, evaluate, ou both.")
    parser.add_argument("--models_path", type=str,
                        help="Chemin des modèles (requis pour parse et both).")
    parser.add_argument("--input_files_path", type=str,
                        help="Chemin des fichiers d'entrée (requis pour parse et both).")
    parser.add_argument("--gold_dir", type=str, help="Chemin des fichiers gold (requis pour evaluate).")
    parser.add_argument("--system_dir", type=str, help="Chemin des fichiers système (requis pour evaluate).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Chemin du répertoire de sortie.")

    args = parser.parse_args()
    main(args)

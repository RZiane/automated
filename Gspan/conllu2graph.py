from conllu import parse

def convert_conllu_to_custom_format(conllu_file_path, output_file_path):
    with open(conllu_file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    sentences = parse(data)

    # Define mappings for UPOS and DEPREL labels to numbers

    upos_mapping = {
        "ADJ": "a", "ADP": "b", "ADV": "c", "AUX": "d",
        "CCONJ": "e", "DET": "f", "INTJ": "g", "NOUN": "h",
        "NUM": "i", "PART": "j", "PRON": "k", "PROPN": "l",
        "PUNCT": "m", "SCONJ": "n", "VERB": "o", "X": "p"
    }

    deprel_mapping = {
        "acl": 1, "advcl": 2, "advmod": 3, "amod": 4,
        "appos": 5, "aux": 6, "case": 7, "cc": 8,
        "ccomp": 9, "clf": 10, "compound": 11, "conj": 12,
        "cop": 13, "csubj": 14, "dep": 15, "det": 16,
        "discourse": 17, "dislocated": 18, "expl": 19, "fixed": 20,
        "flat": 21, "goeswith": 22, "iobj": 23, "list": 24,
        "mark": 25, "nmod": 26, "nsubj": 27, "nummod": 28,
        "obj": 29, "obl": 30, "orphan": 31, "parataxis": 32,
        "punct": 33, "reparandum": 34, "root": 35, "vocative": 36,
        "xcomp": 37
    }


    graphs = []
    for sentence in sentences:
        graph_lines = []

        for token in sentence:
            node_id = token['id']
            upos_label = token['upos']
            upos_code = upos_mapping.get(upos_label, 0)  # Use 0 if not found
            graph_lines.append(f'v {node_id} {upos_label}\n')

        for token in sentence:
            node_id = token['id']
            head_id = token['head']
            deprel_label = token['deprel']
            deprel_code = deprel_mapping.get(deprel_label, 0)  # Use 0 if not found
            if head_id != '_':
                graph_lines.append(f'e {head_id} {node_id} {deprel_label}\n')

        graphs.append(graph_lines)

    # Write to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for i, graph in enumerate(graphs, start=1):
            print(f"t # {i}\n")
            output_file.write(f"t # {i}\n")
            output_file.write(f"v 0 0\n")
            for line in graph:
                print(line, end='')  # Add end='' to avoid extra newlines
                output_file.write(line)

# Example usage
conllu_file_path = '/home/ziane212/projects/AUTOMATED/data/dump_phraseoroche/gold/gold_phraseoroche.conllu'
output_file_path = '/home/ziane212/projects/AUTOMATED/data/dump_phraseoroche/gold/gold_phraseoroche.txt'
convert_conllu_to_custom_format(conllu_file_path, output_file_path)

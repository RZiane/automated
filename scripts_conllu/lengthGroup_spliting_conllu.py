import sys
from utils import def_args
import conllu

def sort_by_length(sent):
    return int(len(sent))

def order_sent_size(inputfile):

    group_1 = []
    group_2 = []
    group_3 = []
    group_4 = []
    group_5 = []
    group_6 = []
    group_7 = []
    group_8 = []
    group_9 = []
    group_10 = []
    group_11 = []

    with open(inputfile, 'r', encoding='utf-8') as infile:
        sentences = list(conllu.parse(infile.read()))
    new_file = []

    for sent in sentences:
        len_sent = 0
        for token in sent:
            len_sent += 1
        
        sent.metadata["len"] = len_sent

        if len_sent < 10:
            group_1.append(sent)
            group_1[:] = sorted(group_1, key=sort_by_length)
        elif len_sent >= 10 and len_sent < 20:
            group_2.append(sent)
            group_2[:] = sorted(group_2, key=sort_by_length)
        elif len_sent >= 20 and len_sent < 30:
            group_3.append(sent)
            group_3[:] = sorted(group_3, key=sort_by_length)
        elif len_sent >= 30 and len_sent < 40:
            group_4.append(sent)
            group_4[:] = sorted(group_4, key=sort_by_length)
        elif len_sent >= 40 and len_sent < 50:
            group_5.append(sent)
            group_5[:] = sorted(group_5, key=sort_by_length)
        elif len_sent >= 50 and len_sent < 60:
            group_6.append(sent)
            group_6[:] = sorted(group_6, key=sort_by_length)
        elif len_sent >= 60 and len_sent < 70:
            group_7.append(sent)
            group_7[:] = sorted(group_7, key=sort_by_length)
        elif len_sent >= 70 and len_sent < 80:
            group_8.append(sent)
            group_8[:] = sorted(group_8, key=sort_by_length)
        elif len_sent >= 80 and len_sent < 100:
            group_9.append(sent)
            group_9[:] = sorted(group_9, key=sort_by_length)
        elif len_sent >= 100 :
            group_10.append(sent)
            group_10[:] = sorted(group_10, key=sort_by_length)
        # elif len_sent >= 90 and len_sent < 100:
        #     group_10.append(sent)
        #     group_10[:] = sorted(group_10, key=sort_by_length)
        # elif len_sent >= 100:
        #     group_11.append(sent)
        #     group_11[:] = sorted(group_11, key=sort_by_length)

    new_file.append(group_1)
    new_file.append(group_2)
    new_file.append(group_3)
    new_file.append(group_4)
    new_file.append(group_5)
    new_file.append(group_6)
    new_file.append(group_7)
    new_file.append(group_8)
    new_file.append(group_9)
    new_file.append(group_10)
    # new_file.append(group_11)

        
    for id_, group in enumerate(new_file):
        output_file_path = inputfile[:-len(".conllu")]+"_"+str(id_)+".conllu"
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # Utiliser writelines pour écrire chaque ligne séparément
            outfile.writelines([sentence.serialize() + "" for sentence in group])


if __name__ == "__main__":
   inputfile, x = def_args(sys.argv[1:])

   order_sent_size(inputfile)
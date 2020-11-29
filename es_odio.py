import os
import sys
from es_odio_tensor_flow import es_odio_TF
import train
import evaluate

def chequear_archivos_impartidos(path):
    # Chequeo de los archivos impartidos (train.csv, val.csv, test.csv fasttext.es.300.txt) en la ruta

    print(f"Check archivos impartidos\n\ttrain.csv\tval.csv\ttest.csv\tfasttext.es.300.txt:")
    train_exists = os.path.isfile(path + '/train.csv')
    val_exists = os.path.isfile(path + '/val.csv')
    test_exists = os.path.isfile(path + '/test.csv')
    fasttext_exists = os.path.isfile(path + '/fasttext.es.300.txt')
    print(f"\t{train_exists}\t{val_exists}\t{test_exists}\t{fasttext_exists}")
    all_file_exist=train_exists and val_exists and test_exists and fasttext_exists
    print(f"\nAll archivos impartidos exist:{all_file_exist}")
    return all_file_exist


def chequear_csvs(path, csvs):
    all_csv_exist = True
    print(f"Check archivos impartidos\n\t{csvs}:", end="\n\t")
    for csv in csvs:
        csv_exists = os.path.isfile(path + '/' + csv)
        print(f"{csv_exists}", end="\t")
        all_csv_exist = all_csv_exist and csv_exists
    print(f"\nAll csv exist:{all_csv_exist}")
    return all_csv_exist


if len(sys.argv) < 3:
    raise ValueError("Debe pasar una ruta a los datos impartidos y al menos un archivo de testeo\n"
          "Ej:  es_odio.py <data_path> test_file1.csv ... test_fileN.csv")
else:
    path, csvs = sys.argv[1], sys.argv[2:]
    if chequear_archivos_impartidos(path):
        if chequear_csvs(path, csvs):
            #Aca deberiamos entrenar el modelo una sola ves.
            model, t, max_length = train.train(path)
            print(f"Path {path}")
            for csv in csvs:
                print(f"\tcsv: {csv}")
                
                evaluate.evaluate(model, t, max_length, path, csv)

                # Aca deberiamos convertir los csv en numpys?
                # Aca deberiamos usar el model para predecir cada csv

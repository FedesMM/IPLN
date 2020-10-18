import pandas as pd
import numpy as np
import re

def load_fasttext_es_300():
    '''Devuelve un numpy array con la palabra y su vector embevido'''
    data = pd.read_csv('fasttext.es.300.txt', header=None, sep=' ')
    data = np.array(data)
    return data[:, :300]

def load_test():
    '''Devuelve un numpy array con la palabra y su vector embevido'''
    data = pd.read_csv('test.csv', header=None, sep='\t')
    data = np.array(data)
    #print(data[:5])
    return data[:, :]


def preprocesing(lines):

    #Usuarios
    #URL

    #print(lines)
    lines = np.array([re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        '@URL', line, flags=re.MULTILINE)
                      for line in lines])
    #print(lines)

    #Multiletra
    #Numeros

    return lines

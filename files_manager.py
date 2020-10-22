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

def load_file(filename):
    data = pd.read_csv(filename, header=None, sep='\t')
    data = np.array(data)
    return data[:, :]


def preprocesing(lines):


    print(lines)

    #Usuarios
    lines = np.array([re.sub(
        r'''(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)''',
        '@USUARIOS', line, flags=re.MULTILINE)
        for line in lines])
    #URL
    lines = np.array([re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        '', line, flags=re.MULTILINE)
                      for line in lines])

    # Hashtags
    lines = np.array([re.sub(
        r'''\#''',
        '', line, flags=re.MULTILINE)
                      for line in lines])
    #Multiletra
    lines = np.array([re.sub(
        r'''(\w)\1{2,}''',
        r'''\1''', line, flags=re.MULTILINE)
        for line in lines])
    print(lines)
    #Numeros

    #Lower se maneja desde el tokenize
    #lines = np.char.lower(lines)

    return lines

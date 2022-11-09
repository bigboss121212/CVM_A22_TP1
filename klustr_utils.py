

import numpy as np
from PySide6 import QtCore, QtGui
from __feature__ import snake_case, true_property 


def qimage_argb32_from_png_decoding(img_data):
    '''Effectue le décodage d'un 'buffer' de données correspondant à une 
       image PNG. 

       En entrée se trouve le 'buffer' conforme d'une image de format PNG.

       En sortie on obtient une QImage du format QImage.Format_ARGB32.'''
    image = QtGui.QImage()
    if image.load_from_data(QtCore.QByteArray(bytearray(img_data)), 'png'):
        return image.convert_to_format(QtGui.QImage.Format_ARGB32)
    
    print("Erreur de décodage d'une image avec la fonction _png_decoding.")
    return image

def ndarray_from_qimage_argb32(img):
    '''Effectue la conversion d'une image de format 
       QImage.Format_ARGB32 vers une matrice numpy.

       L'image d'entrée doit respectée le format spécifié et correspondre 
       à une image binaire avec les couleurs noir et blanc.

       L'image de sortie est convertie en une matrice de la même taille ayant des valeurs 0-1 en format uint8.'''
    return (np.frombuffer(img.bits(), dtype=np.uint32).reshape((img.height(), img.width())) != 0xFF000000).astype(np.uint8)
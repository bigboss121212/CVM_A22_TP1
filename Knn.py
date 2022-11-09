import numpy as np
import klustr_utils as util

class Knn():

    def __init__(self):
        self.matrice3d_descripteur = None

    def initMatrice(self, value):
        self.matrice3d_descripteur = np.arange(value * 4).astype(np.float32)
        self.matrice3d_descripteur = np.resize(self.matrice3d_descripteur, ((value, 4)))
        self.matrice3d_descripteur[:] = 0

    def decodeImage(self, value):

        matrice_training = util.qimage_argb32_from_png_decoding(value[0][0])
        matrice_training = util.ndarray_from_qimage_argb32(matrice_training)
        matrice_training = matrice_training + 1
        matrice_training[matrice_training == 2] = 0
        return matrice_training

    def evaluation_training(self, value):

        self.initMatrice(value.__len__())

        for j in range(value.__len__()):
            matrice_training = util.qimage_argb32_from_png_decoding(value[j][6])
            matrice_training = util.ndarray_from_qimage_argb32(matrice_training)
            matrice_training = matrice_training + 1
            matrice_training[matrice_training == 2] = 0
            descripteur_1 = self.descripteur1(matrice_training)
            descripteur_2 = self.descripteur2(matrice_training)
            descripteur_3 = self.descripteur3(matrice_training)

            descripteur_training = np.array([descripteur_1, descripteur_2, descripteur_3, value[j][0]])
            self.matrice3d_descripteur[j] = descripteur_training

        return self.matrice3d_descripteur

    def evaluation_test(self, value):

        self.initMatrice(1)
        matrice_training = self.decodeImage(value)

        descripteur_1 = self.descripteur1(matrice_training)
        descripteur_2 = self.descripteur2(matrice_training)
        descripteur_3 = self.descripteur3(matrice_training)

        descripteur_training = np.array([descripteur_1, descripteur_2, descripteur_3])

        return descripteur_training

    def air(self, matrice):
        return np.sum(matrice[matrice == 1])

    def descripteur1(self, matrice):
        air = self.air(matrice)
        # pour trouver le perimetre tres fortement inspirer de
        # https://stackoverflow.com/questions/13443246/calculate-perimeter-of-numpy-array
        per = np.sum(matrice[:, 1:] != matrice[:, :-1]) + np.sum(
            matrice[1:, :] != matrice[:-1, :])
        return (4 * np.pi * air) / per**2

    def descripteur2(self, matrice):
        rayon = self.rayon_min_max(matrice, True)
        air_cercle_circonscrit = rayon * rayon * np.pi
        return self.air(matrice) / air_cercle_circonscrit

    def descripteur3(self, matrice):
        rayon1 = self.rayon_min_max(matrice, False)
        air_cercle_inscrit = rayon1 * rayon1 * np.pi
        rayon2 = self.rayon_min_max(matrice, True)
        air_cercle_circonscrit = rayon2 * rayon2 * np.pi
        return air_cercle_inscrit/ air_cercle_circonscrit

    def rayon_min_max(self, matrice, bool):
        voisin = np.zeros(matrice.shape, dtype=np.float64)
        centro = self.centroid(matrice)

        # pour trouver les coordonnees du contour inspirer de code sur internet, lien perdu mais bonne
        # comprehension du code
        voisin[1:] += matrice[:-1]  # Nord
        voisin[:-1] += matrice[1:]  # Sud
        voisin[:, :-1] += matrice[:, 1:]  # Est
        voisin[:, 1:] += matrice[:, :-1]  # Ouest
        voisin[voisin != 4] = 0
        voisin[voisin == 4] = 1
        point_exterieur = matrice - voisin
        coordo_exterieur = np.where(point_exterieur == 1)
        nbr_coordo = len(coordo_exterieur[0])

        c, r = np.meshgrid(np.arange(matrice.shape[1]), np.arange(matrice.shape[0]))
        points = np.empty((nbr_coordo, 2), np.int)  ##pour rentrer les coordonnes

        points[:, 1] = c[point_exterieur == 1]
        points[:, 0] = r[point_exterieur == 1]

        result = points - centro  ##euclidienne
        result = result ** 2
        result = result[:, :1] + result[:, 1:2]
        result = result ** 0.5
        if(bool == True):
            dist = np.max(result)
        else:
            dist = np.min(result)
        return dist

    def area(self, image):
        return np.sum(image)

    def centroid(self, image):
        c, r = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        return (np.sum(r * image), np.sum(c * image)) / self.area(image)

    def trouver_forme_test(self, matrice_desc_test, matrice3d_descripteur, k = 3):

        matrice_desc_t = np.array([matrice_desc_test])
        self.matrice3d_descripteur = matrice3d_descripteur

        cdistance1 = self.matrice3d_descripteur[:, 0:3] - matrice_desc_t
        cdistance2 = cdistance1 ** 2
        distance = cdistance2[:, :1] + cdistance2[:, 1:2] + cdistance2[:, 2:3]
        distanceCalcule = distance ** 0.5

        dist_n_label = np.concatenate((distanceCalcule, self.matrice3d_descripteur[:, -1:]), axis=1)

        ind = np.lexsort((dist_n_label[:, 1], dist_n_label[:, 0])) #compris mais trouvé sur https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        distance_trie = dist_n_label[ind]
        points_k = distance_trie[:k,:]

        data_final = points_k[:, 1].flatten()
        data_final.astype(np.int64)
        # pour sortir la valeur qui sera la plus recurente dans cette matrice, je concidere que ce sera
        # necessairement un label
        label = np.bincount(data_final.astype(np.int64)).argmax()
        return label

def main():
    k = Knn()

if __name__ == '__main__':
    quit(main())
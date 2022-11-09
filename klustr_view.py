from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QRect
from PySide6.QtGui import qRgb, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QScrollBar, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QGroupBox, QComboBox, \
    QMessageBox
import sys
import Knn as k
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from db_credential import PostgreSQLCredential
from klustr_dao import PostgreSQLKlustRDAO
import klustr_utils


class ScatterDiagram:
    def __init__(self):
        self.__figure = plt.figure(figsize=(5, 5))
        self.__axes = plt.axes(projection='3d')
        self.__canvas = FigureCanvas(self.__figure)
        self.__canvas.draw()
        self.init_ui_graph()

    def init_ui_graph(self):
        plt.cla()  # pour clear sans le fermer
        self.__axes.set_xlabel('X-axis', fontweight='bold')
        self.__axes.set_ylabel('Y-axis', fontweight='bold')
        self.__axes.set_zlabel('Z-axis', fontweight='bold')
        self.__axes.set_title("Graphique de dispersion")

    def update_data(self, matrice):
        self.init_ui_graph()
        x_vals = matrice[:, 0:1]
        y_vals = matrice[:, 1:2]
        z_vals = matrice[:, 2:3]
        self.__axes.scatter3D(x_vals, y_vals, z_vals, color='blue', marker="o")
        plt.draw()

    def update_point_test(self, matrice):
        x_vals = matrice[0]
        y_vals = matrice[1]
        z_vals = matrice[2]
        self.__axes.scatter3D(x_vals, y_vals, z_vals, color='red', marker="^")
        plt.draw()

    @property
    def widget(self):
        return self.__canvas


class KlustR(QtWidgets.QMainWindow):
    def __init__(self, klustr_dao, knn, parent=None):
        super().__init__(parent)
        self.klustr_dao = klustr_dao
        self.displayPng = QPixmap(klustr_utils.qimage_argb32_from_png_decoding(
            self.klustr_dao.get_image_test("img_ellipsoid_200_200_100_0031")[0][0]))
        self.label_displayPng = QLabel()

        self.dataset_stats = QVBoxLayout()
        self.dataset_transform_stats = QVBoxLayout()
        self.k = 1

        self.comboBox_datasetNames = None
        self.classified_status_label = QLabel()
        self.setWindowTitle("KlustR KNN Classifier")
        self.knn = knn
        self.main_window_widget = QWidget()
        self.main_window_layout = QHBoxLayout()
        self.dataset_gb = self.data_set_groupbox()
        self.comboBox_datasetsLabels = self.dataset_gb.create_drop_list(self.update_dataset_labels())
        self.single_test_gb = self.single_test_groupbox()
        self.knn_params_gb = self.knn_parameters_groupbox()
        self.about_button = QPushButton("About")

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.dataset_gb)
        control_layout.addWidget(self.single_test_gb)
        control_layout.addWidget(self.knn_params_gb)
        control_layout.addWidget(self.about_button)
        self.main_window_layout.addLayout(control_layout)
        self.__scatter = ScatterDiagram()
        self.main_window_layout.addWidget(self.graphique3D())
        self.main_window_widget.setLayout(self.main_window_layout)
        self.setCentralWidget(self.main_window_widget)
        self.matrice_descripteur_training = self.knn.evaluation_training(
            self.klustr_dao.get_dataset_tests(self.data_set_current_name()))
        self.__scatter.update_data(self.matrice_descripteur_training)
        self.update_dataset_stats()
        self.about_button.clicked.connect(self.about_show)

    def datasets_available(self):
        data = []
        for dataset in self.klustr_dao.available_datasets:
            data.append(str(dataset[1]) + " " + "[" + str(dataset[5]) + "]" + "[" + str(dataset[8]) + "]")
        return data

    def update_dataset_labels(self):
        data = []
        for dataset in self.klustr_dao.image_from_dataset(self.data_set_current_name(), False):
            data.append(str(dataset[3]))
        return data

    def data_set_groupbox(self):
        group_box = TemplateGB("Dataset", 390, 180)
        big_layout = QVBoxLayout()  # main layout couvre tout groupbox
        self.comboBox_datasetNames = group_box.create_drop_list(self.datasets_available())
        self.comboBox_datasetNames.currentIndexChanged.connect(self.update_dataset_stats)  # nom du data set
        big_layout.addWidget(self.comboBox_datasetNames)  # ajout select bar
        dataset_infos_layout = QHBoxLayout()  # layout interieur en bas pour les 2 groupbox
        dataset_infos_layout.addWidget(self.included_in_dataset())
        dataset_infos_layout.addWidget(self.transformation())
        # ajout fin
        big_layout.addLayout(dataset_infos_layout)
        group_box.setLayout(big_layout)
        return group_box

    def included_in_dataset(self):
        group_box = QtWidgets.QGroupBox("Included in dataset")
        big_layout = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(QLabel("Category count:"))
        titles.addWidget(QLabel("Training image count:"))
        titles.addWidget(QLabel("Test image count:"))
        titles.addWidget(QLabel("Total image count:"))

        big_layout.addLayout(titles)
        big_layout.addLayout(self.dataset_stats)
        group_box.setLayout(big_layout)
        return group_box

    def transformation(self):
        group_box = QtWidgets.QGroupBox("Transformation")
        big_layout = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(QLabel("Translated:"))
        titles.addWidget(QLabel("Rotated:"))
        titles.addWidget(QLabel("Scaled:"))
        titles.addStretch()
        big_layout.addLayout(titles)
        big_layout.addLayout(self.dataset_transform_stats)
        group_box.setLayout(big_layout)
        return group_box

    def single_test_groupbox(self):
        group_box = TemplateGB("Single test", 390, 290)
        big_layout = QVBoxLayout()
        self.comboBox_datasetsLabels.currentIndexChanged.connect(self.update_data_png)
        big_layout.addWidget(self.comboBox_datasetsLabels)  # ajout select bar
        image_layout = QVBoxLayout()
        self.label_displayPng.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_displayPng.setPixmap(self.displayPng)
        self.label_displayPng.setFixedHeight(170)
        self.label_displayPng.setStyleSheet("background:#223544;")
        self.label_displayPng.alignment = Qt.AlignCenter
        image_layout.addWidget(self.label_displayPng)
        classify_button = QPushButton("Classify")
        classify_button.setFixedHeight(25)
        classify_button.clicked.connect(self.classify_forme_test)  # ajout Alex pour classifier la forme test
        self.classified_status_label = QLabel("not classified")
        self.classified_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        big_layout.addLayout(image_layout)
        big_layout.addWidget(classify_button)
        big_layout.addWidget(self.classified_status_label)
        group_box.setLayout(big_layout)

        return group_box

    def knn_parameters_groupbox(self):
        group_box = TemplateGB("Knn parameters", 390, 80)
        big_layout = QHBoxLayout()
        mini_layout = QHBoxLayout()
        mini_layout.addWidget(QLabel("K ="))
        data = QLabel('1')
        mini_layout.addWidget(data)
        max_dis_select_layout = QHBoxLayout()
        title_max_dist = QLabel("Max dist =")
        title_max_dist.setFixedWidth(60)
        max_dis_select_layout.addWidget(title_max_dist)
        max_data = QLabel("0.01")
        max_data.setFixedWidth(25)
        max_dis_select_layout.addWidget(max_data)
        titles_layout = QVBoxLayout()
        titles_layout.addLayout(mini_layout)
        titles_layout.addLayout(max_dis_select_layout)
        scroll_layout = QVBoxLayout()
        knn_scroll = QScrollBar()
        knn_scroll.setMinimum(1)
        knn_scroll.setMaximum(3)
        knn_scroll.valueChanged.connect(lambda: self.change_k_value(data, knn_scroll))
        knn_scroll.setOrientation(Qt.Horizontal)
        knn_scroll.setFixedWidth(270)
        dist_max_scroll = QScrollBar()
        dist_max_scroll.setMinimum(0.01)
        dist_max_scroll.setMaximum(99.9)
        dist_max_scroll.setOrientation(Qt.Horizontal)
        dist_max_scroll.setFixedWidth(270)
        scroll_layout.addWidget(knn_scroll)
        scroll_layout.addWidget(dist_max_scroll)
        big_layout.addLayout(titles_layout)
        big_layout.addLayout(scroll_layout)
        big_layout.addStretch()
        group_box.setLayout(big_layout)
        return group_box

    def change_k_value(self, data, k):
        data.setNum(k.value())
        self.k = k.value()

    def graphique3D(self):
        return self.__scatter.widget

    def update_data_png(self):
        if self.comboBox_datasetsLabels.currentText().split():
            val = (self.comboBox_datasetsLabels.currentText().split())[0]
            img = klustr_utils.qimage_argb32_from_png_decoding(self.klustr_dao.get_image_test(val)[0][0])
            self.label_displayPng.setPixmap(QPixmap(img))

    def update_dataset_stats(self):
        requete_image = self.klustr_dao.get_dataset_tests(self.data_set_current_name())
        self.matrice_descripteur_training = self.knn.evaluation_training(requete_image)
        self.comboBox_datasetsLabels.clear()
        self.comboBox_datasetsLabels.addItems(self.update_dataset_labels())
        self.__scatter.update_data(self.matrice_descripteur_training)
        tab = self.klustr_dao.get_current_dataset_info(self.data_set_current_name())

        for i in reversed(range(self.dataset_stats.count())):
            self.dataset_stats.itemAt(i).widget().deleteLater()
        for j in reversed(range(self.dataset_transform_stats.count())):
            self.dataset_transform_stats.itemAt(j).widget().deleteLater()

        self.dataset_stats.addWidget(QLabel(str(tab[0][5])))
        self.dataset_stats.addWidget(QLabel(str(tab[0][6])))
        self.dataset_stats.addWidget(QLabel(str(tab[0][7])))
        self.dataset_stats.addWidget(QLabel(str(tab[0][8])))

        self.dataset_transform_stats.addWidget(QLabel(str(tab[0][2])))
        self.dataset_transform_stats.addWidget(QLabel(str(tab[0][3])))
        self.dataset_transform_stats.addWidget(QLabel(str(tab[0][4])))

    def data_set_current_name(self):
        return (self.comboBox_datasetNames.currentText().split())[0]

    def data_set_current_name_test(self):
        return self.comboBox_datasetsLabels.currentText()

    def classify_forme_test(self):
        requete_image = self.klustr_dao.get_image_test(self.data_set_current_name_test())
        descripteur_test = self.knn.evaluation_test(requete_image)
        self.__scatter.update_point_test(descripteur_test)
        forme_trouve = self.knn.trouver_forme_test(descripteur_test, self.matrice_descripteur_training, self.k)
        nom_label_test = self.klustr_dao.get_label_test(str(int(forme_trouve)))
        self.classified_status_label.setText(nom_label_test[0][0])

    def about_show(self):
        data = """
        Ce logiciel est le projet no 1 du cours C52.

        Il a été réalisé par : 
        - Roberto N.
        - Jeremie K.
        - Antonin L.
        - Alexandre C.

        Il consiste à faire __quelque_chose__ avec les concepts suivants :
        - Machine learning 
        - la classification et la régression

        Nos 3 descripteurs de forme sont :
        - La complexité : 
         - Consiste à calculer l'air et le perimètre de la forme. On normalise la 
         complexité.
        - Le ratio entre le cercle circonscrit et l'air. 
         - Notre calcul nous donnera un ratio pour chaque forme et n'est pas sujet 
         à la translation, la rotationn ou le scale 
        - Le ratio entre le cercle inscrit et le cercle circonscrit 
         - Notre calcul nous donnera un ratio pour chaque forme et n'est pas sujet
         à la translation, la rotationn ou le scale

        Plus précisément, ce laboratoire permet de mettre en pratique les notions de : 
        - Traitement de données
        - Intéragir avec une base de données contenant des milliers de données
        - Optimisation de calculs sans boucle for avec Numpy
        - Optimisation de l'exécution du temps d'affichage et le traitement des données
        - Travail collaboratif

        Un effort d'abstraction a été fait pour ces points : 
        - La classe KNN
        - Le klustr View

        Finalement, l’ensemble de données le plus complexe que nous avons été capable 
        de résoudre est:
        - Zoo-Large

        """
        detail = QMessageBox.about(self, "KlustR KNN Classifier", data)


class TemplateGB(QtWidgets.QGroupBox):
    def __init__(self, title, width, height, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        self.setFixedWidth(width)
        self.setFixedHeight(height)
        self.layout()

    def create_drop_list(self, data):
        cmbbox_widget = QComboBox(self)
        drop_list_content = data
        cmbbox_widget.addItems(drop_list_content)
        return cmbbox_widget


def main():
    app = QtWidgets.QApplication(sys.argv)
    credential = PostgreSQLCredential(password='ASDasd123')
    klustr_dao = PostgreSQLKlustRDAO(credential)
    klustR = KlustR(klustr_dao, k.Knn())
    klustR.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

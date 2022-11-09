from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QRect
from PySide6.QtGui import qRgb, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QScrollBar, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QGroupBox, QComboBox
import sys

import numpy as np

from connexionPGSQL import DataBase
from db_credential import PostgreSQLCredential
from klustr_dao import PostgreSQLKlustRDAO
from klustr_utils import qimage_argb32_from_png_decoding
from __feature__ import snake_case, true_property


class KlustR(QtWidgets.QMainWindow):
    def __init__(self, klustr_dao, parent=None):
        super().__init__(parent)

        self.klustr_dao = klustr_dao

        self.set_window_title("KlustR KNN Classifier")
        self.__graph_label = QLabel()
        self.__graph_label.alignment = Qt.AlignCenter

        self.connexionPGSQL = DataBase()
        self.__dataset_gb = Data_set_Group_Box("Dataset", self.connexionPGSQL.getInfos())
        self.__single_test_gb = self.createSecondGroupBox()
        self.__knn_params_gb = self.createThirdGroupBox()

        image_data_info_layout = QVBoxLayout()
        image_data_info_layout.add_widget(self.__dataset_gb)
        image_data_info_layout.add_widget(self.__single_test_gb)
        image_data_info_layout.add_widget(self.__knn_params_gb)

        main_window_layout = QHBoxLayout()
        main_window_layout.add_layout(image_data_info_layout)
        main_window_layout.add_widget(self.__graph_label)

        central_widget = QWidget()
        central_widget.set_layout(main_window_layout)

        self.set_central_widget(central_widget)
        self.__create_template_image()

    # ---------------------------------------------------------------------------------------------------------------------
    def createSecondGroupBox(self):
        qgroupBox = QtWidgets.QGroupBox("Single test")

        combo_box = QComboBox(self)
        resolution_list = ["img_ellipsoid_200_200_100_0031", "200x200", "300x300", "400x400", "500x500"]
        combo_box.add_items(resolution_list)

        labelimage = QPixmap("images\download.png")
        label = QLabel()
        label.set_pixmap(labelimage);
        label.set_fixed_width(200)
        label.set_fixed_height(200)

        stop_button = QPushButton("Classify")

        level = QLabel("not classified")
        level.alignment = Qt.AlignCenter
        layoutV = QVBoxLayout()
        layoutV.add_widget(combo_box)
        layoutV.add_widget(label)
        layoutV.add_widget(stop_button)
        layoutV.add_widget(level)
        layoutV.add_stretch(1)
        qgroupBox.set_fixed_height(300)
        qgroupBox.set_fixed_width(400)
        qgroupBox.set_layout(layoutV)

        return qgroupBox

    def createThirdGroupBox(self):
        qgroupBox = QtWidgets.QGroupBox("Knn parameters")

        layoutH = QHBoxLayout()
        layoutH2 = QHBoxLayout()

        k = QLabel("K = ")
        kn = QLabel()
        max = QLabel("Max dist = ")
        maxD = QLabel()
        scroll = QScrollBar()
        scroll.set_orientation(Qt.Horizontal)
        scroll.set_fixed_width(300)
        scrollVitesse = QScrollBar()
        scrollVitesse.set_orientation(Qt.Horizontal)
        scrollVitesse.set_fixed_width(300)

        layoutH.add_widget(k)
        layoutH.add_widget(kn)
        layoutH.add_widget(scroll)

        layoutH2.add_widget(max)
        layoutH2.add_widget(maxD)
        layoutH2.add_widget(scrollVitesse)

        layoutV = QVBoxLayout()
        # layoutV.add_widget(labelimage)
        layoutV.add_layout(layoutH)
        layoutV.add_layout(layoutH2)

        layoutV.add_stretch(1)
        qgroupBox.set_fixed_height(100)
        qgroupBox.set_fixed_width(400)
        qgroupBox.set_layout(layoutV)

        return qgroupBox

    def __create_template_image(self):
        image = QtGui.QImage(100, 100, QtGui.QImage.Format_ARGB32)
        for y in range(0, 100):
            for x in range(0, 100):
                image.set_pixel_color(x, y, QtGui.QColor(0, 0, 0))
        pixmap = QtGui.QPixmap.from_image(image.scaled(500, 500))
        self.__graph_label.set_pixmap(pixmap)

    def chercher_donnees(self):
        data = []
        for dataset in self.klustr_dao.available_datasets:
            data.append(str(dataset[1]) + " " + "[" + str(dataset[5]) + "]" + "[" + str(dataset[8]) + "]")
        self.closeConnection()
        return data


class Data_set_Group_Box(QtWidgets.QGroupBox):
    def __init__(self, title, data, parent=None):
        super().__init__(parent)
        self.title = title
        self.set_fixed_width(390)
        self.set_fixed_height(180)
        ##
        drop_list_widget = QComboBox(self)#
        self.drop_list_content = data#
        drop_list_widget.add_items(self.drop_list_content)#
        #
        data_set_central_layout = QVBoxLayout()
        data_set_central_layout.add_widget(drop_list_widget)
        #
        information_groupboxes_layout = QHBoxLayout()
        #
        data_set_groupbox = QtWidgets.QGroupBox("Included in dataset")

        titles_layout = QVBoxLayout()
        titles_layout.add_widget(QLabel("Category count:"))
        titles_layout.add_widget(QLabel("Training image count:"))
        titles_layout.add_widget(QLabel("Test image count:"))
        titles_layout.add_widget(QLabel("Total image count:"))
        #
        info_stats = QVBoxLayout()
        info_stats.add_widget(QLabel("2"))
        info_stats.add_widget(QLabel("4"))
        info_stats.add_widget(QLabel("6"))
        info_stats.add_widget(QLabel("10"))

        info_layout = QHBoxLayout()
        info_layout.add_layout(titles_layout)
        info_layout.add_layout(info_stats)

        data_set_groupbox.set_layout(info_layout)


        #
        transformation_groupbox = QtWidgets.QGroupBox("Transformation")

        information_groupboxes_layout.add_widget(data_set_groupbox) #ajout included gb
        information_groupboxes_layout.add_widget(transformation_groupbox) #ajout transforma gb
        #
        data_set_central_layout.add_layout(information_groupboxes_layout)
        ##
        self.set_layout(data_set_central_layout)
        ###

def main():
    app = QtWidgets.QApplication(sys.argv)
    credential = PostgreSQLCredential(password='ASDasd123')
    klustr_dao = PostgreSQLKlustRDAO(credential)
    klustR = KlustR(klustr_dao)
    klustR.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

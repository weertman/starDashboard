#dashboard.py

"""
Photo Re-Identification Tool

This script implements a GUI application for comparing and analyzing animal photos
across multiple directories. It uses PyQt6 for the interface and allows users to
manipulate images for detailed comparison.

Usage: python photo_reidentification_tool.py [list of directories] [list of codes]
"""

import sys
import os
import glob
from typing import List, Dict
import logging
from datetime import datetime
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QPushButton, QComboBox, QFileDialog, QTextEdit,
                             QLineEdit, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap, QTransform, QPainter, QWheelEvent, QMouseEvent
from PyQt6.QtCore import Qt, QPointF, QDir
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class UnderwaterColorCorrection:
    def __init__(self, image):
        self.original_image = image.astype(np.float32) / 255.0
        self.avg_color = self.calculate_average_color()
        self.original_avg_red = self.avg_color[0]

    def calculate_average_color(self):
        return np.mean(self.original_image, axis=(0, 1))

    def apply_correction(self, adjustment):
        # Adjustment range: -100 to 100
        # Convert to a factor: 0.5 to 1.5
        factor = 1 + (adjustment / 200)

        corrected = self.original_image.copy()

        # Adjust red channel
        corrected[:, :, 0] = np.clip(corrected[:, :, 0] * factor, 0, 1)

        # Slightly adjust green and blue to maintain color balance
        if factor > 1:
            corrected[:, :, 1] = np.clip(corrected[:, :, 1] * (1 + (factor - 1) * 0.2), 0, 1)
            corrected[:, :, 2] = np.clip(corrected[:, :, 2] * (1 + (factor - 1) * 0.1), 0, 1)
        else:
            corrected[:, :, 1] = np.clip(corrected[:, :, 1] * (1 + (factor - 1) * 0.1), 0, 1)
            corrected[:, :, 2] = np.clip(corrected[:, :, 2] * (1 + (factor - 1) * 0.2), 0, 1)

        # Convert back to uint8
        return (corrected * 255).astype(np.uint8)

class ComparisonLogger:
    def __init__(self, src_data_dir):
        self.csv_path = os.path.join(src_data_dir, 'comparison_results.csv')
        self.columns = ['DATE', 'TIME', 'ID_A', 'ID_B', 'ARE_SAME', 'FULL_PATH_A', 'FULL_PATH_B',
                        'ROTATION_A', 'BRIGHTNESS_A', 'CONTRAST_A', 'RED_SHIFT_A',
                        'ROTATION_B', 'BRIGHTNESS_B', 'CONTRAST_B', 'RED_SHIFT_B',
                        'NOTES', 'USER']

    def log_comparison(self, id_a, id_b, are_same, full_path_a, full_path_b,
                       rotation_a, brightness_a, contrast_a, color_correction_a,
                       rotation_b, brightness_b, contrast_b, color_correction_b,
                       notes, user):
        try:
            print(f"Logging comparison: {id_a} vs {id_b}, status={are_same}, user={user}")

            current_datetime = datetime.now()
            current_date = current_datetime.strftime('%d-%m-%Y')
            current_time = current_datetime.strftime('%H-%M-%S')

            try:
                df = pd.read_csv(self.csv_path)
            except FileNotFoundError:
                df = pd.DataFrame(columns=self.columns)

            new_data = {
                'DATE': current_date,
                'TIME': current_time,
                'ID_A': id_a,
                'ID_B': id_b,
                'ARE_SAME': are_same,
                'FULL_PATH_A': full_path_a,
                'FULL_PATH_B': full_path_b,
                'ROTATION_A': rotation_a,
                'BRIGHTNESS_A': brightness_a,
                'CONTRAST_A': contrast_a,
                'RED_SHIFT_A': color_correction_a,
                'ROTATION_B': rotation_b,
                'BRIGHTNESS_B': brightness_b,
                'CONTRAST_B': contrast_b,
                'RED_SHIFT_B': color_correction_b,
                'NOTES': notes,
                'USER': user
            }

            print(f"Debug: Selecting row with ID_A={id_a}, ID_B={id_b}, USER={user}")
            mask = (df['ID_A'] == id_a) & (df['ID_B'] == id_b) & (df['USER'] == user)
            print(f"Debug: Number of rows selected: {mask.sum()}")

            if mask.sum() == 0:
                # No matching row, append new row
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            elif mask.sum() == 1:
                # One matching row, update it
                for column, value in new_data.items():
                    df.loc[mask, column] = value
            else:
                # Multiple matching rows, this shouldn't happen
                raise ValueError(
                    f"Multiple rows ({mask.sum()}) found for ID_A={id_a}, ID_B={id_b}, USER={user}. This indicates a data integrity issue.")

            df['ARE_SAME'] = df['ARE_SAME'].fillna('')

            df.to_csv(self.csv_path, index=False)
            print(f"Comparison logged successfully")
        except Exception as e:
            print(f"Error logging comparison: {str(e)}")
            logging.error(f"Error logging comparison: {str(e)}", exc_info=True)

class ImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image = None
        self.current_image = None
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.rotation_angle = 0
        self.last_pan_pos = None
        self.color_corrector = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create a fixed-size widget to contain the image label
        self.image_container = QWidget()
        self.image_container.setFixedSize(800, 800)
        self.image_container_layout = QVBoxLayout(self.image_container)
        self.image_container_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_container_layout.addWidget(self.image_label)

        layout.addWidget(self.image_container)

        # Brightness slider
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.brightness = QSlider(Qt.Orientation.Horizontal)
        self.brightness.setRange(-100, 100)
        self.brightness.setValue(0)
        brightness_layout.addWidget(self.brightness)
        layout.addLayout(brightness_layout)

        # Contrast slider
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast = QSlider(Qt.Orientation.Horizontal)
        self.contrast.setRange(0, 200)
        self.contrast.setValue(100)
        contrast_layout.addWidget(self.contrast)
        layout.addLayout(contrast_layout)

        # Rotation slider
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation:"))
        self.rotation = QSlider(Qt.Orientation.Horizontal)
        self.rotation.setRange(-180, 180)
        self.rotation.setValue(0)
        rotation_layout.addWidget(self.rotation)
        layout.addLayout(rotation_layout)

        # Red Shift slider
        color_correction_layout = QHBoxLayout()
        color_correction_layout.addWidget(QLabel("Red Shift:"))
        self.color_correction_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_correction_slider.setRange(-100, 100)
        self.color_correction_slider.setValue(0)
        color_correction_layout.addWidget(self.color_correction_slider)
        layout.addLayout(color_correction_layout)

        # Reset button
        self.reset_button = QPushButton("Reset Image")
        layout.addWidget(self.reset_button)

        self.setLayout(layout)

        # Connect signals
        self.brightness.valueChanged.connect(self.update_image)
        self.contrast.valueChanged.connect(self.update_image)
        self.rotation.valueChanged.connect(self.update_image)
        self.color_correction_slider.valueChanged.connect(self.update_image)
        self.reset_button.clicked.connect(self.reset_image)

        # Enable mouse tracking
        self.setMouseTracking(True)

    def load_image(self, image_path: str):
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise IOError(f"Unable to load image: {image_path}")
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.current_image = self.original_image.copy()
            self.zoom_factor = 1.0
            self.pan_offset = QPointF(0, 0)
            self.rotation_angle = 0
            self.color_corrector = UnderwaterColorCorrection(self.original_image)
            self.color_correction_slider.setValue(0)
            self.fit_image_to_container()
            self.update_image()
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}", exc_info=True)
            self.image_label.setText(f"Error loading image: {str(e)}")

    def fit_image_to_container(self):
        if self.current_image is None:
            return

        container_size = self.image_container.size()
        image_size = self.current_image.shape[:2][::-1]  # width, height

        # Calculate the scaling factor to fit the image within the container
        scale_factor = min(container_size.width() / image_size[0],
                           container_size.height() / image_size[1])

        self.zoom_factor = scale_factor

    def update_image(self):
        if self.current_image is None:
            return

        # Apply color correction
        adjustment = self.color_correction_slider.value()
        corrected_image = self.color_corrector.apply_correction(adjustment)

        # Apply other transformations
        transformed = self.apply_transformations(corrected_image)

        # Convert to QPixmap and display
        height, width = transformed.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(transformed.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Create a base pixmap with the size of the container
        base_pixmap = QPixmap(self.image_container.size())
        base_pixmap.fill(Qt.GlobalColor.transparent)

        # Create a painter to draw on the base pixmap
        painter = QPainter(base_pixmap)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Apply zoom, pan, and rotation
        painter.translate(self.image_container.width() / 2 + self.pan_offset.x(),
                          self.image_container.height() / 2 + self.pan_offset.y())
        painter.rotate(self.rotation.value())
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(-pixmap.width() / 2, -pixmap.height() / 2)

        # Draw the transformed image
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        self.image_label.setPixmap(base_pixmap)

    def apply_transformations(self, image):
        # Apply brightness and contrast
        brightness = self.brightness.value()
        contrast = self.contrast.value() / 100.0
        adjusted = cv2.addWeighted(image, contrast, image, 0, brightness)
        return adjusted

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.fit_image_to_container()
            self.pan_offset = QPointF(0, 0)
            self.brightness.setValue(0)
            self.contrast.setValue(100)
            self.rotation.setValue(0)
            self.color_correction_slider.setValue(0)  # Reset color correction
            self.update_image()

    def wheelEvent(self, event: QWheelEvent):
        # Zoom with mouse wheel
        zoom_in_factor = 1.1
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.zoom_factor *= zoom_in_factor
        else:
            self.zoom_factor *= zoom_out_factor

        self.update_image()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pan_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.last_pan_pos:
                delta = event.position() - self.last_pan_pos
                self.pan_offset += delta
                self.last_pan_pos = event.position()
                self.update_image()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pan_pos = None
class MainWindow(QMainWindow):
    def __init__(self, directories: List[str], codes: List[str]):
        super().__init__()
        self.directories = directories
        self.codes = codes
        self.current_images = {"A": [], "B": []}
        self.current_indices = {"A": 0, "B": 0}
        self.comparison_logger = ComparisonLogger(get_src_data_dir())
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel
        left_panel = QVBoxLayout()
        self.panel_a = ImagePanel()
        self.dir_combo_a = QComboBox()
        self.dir_combo_a.addItems(self.codes)
        left_panel.addWidget(self.dir_combo_a)
        left_panel.addWidget(self.panel_a)

        # Create horizontal layout for navigation buttons A
        nav_buttons_layout_a = QHBoxLayout()
        self.prev_button_a = QPushButton("Previous Image A")
        self.next_button_a = QPushButton("Next Image A")
        nav_buttons_layout_a.addWidget(self.prev_button_a)
        nav_buttons_layout_a.addWidget(self.next_button_a)
        left_panel.addLayout(nav_buttons_layout_a)

        main_layout.addLayout(left_panel)

        # Right panel
        right_panel = QVBoxLayout()
        self.panel_b = ImagePanel()
        self.dir_combo_b = QComboBox()
        self.dir_combo_b.addItems(self.codes)
        right_panel.addWidget(self.dir_combo_b)
        right_panel.addWidget(self.panel_b)

        # Create horizontal layout for navigation buttons B
        nav_buttons_layout_b = QHBoxLayout()
        self.prev_button_b = QPushButton("Previous Image B")
        self.next_button_b = QPushButton("Next Image B")
        nav_buttons_layout_b.addWidget(self.prev_button_b)
        nav_buttons_layout_b.addWidget(self.next_button_b)
        right_panel.addLayout(nav_buttons_layout_b)

        main_layout.addLayout(right_panel)

        # Central controls
        control_layout = QVBoxLayout()
        self.same_button = QPushButton("Same Individual")
        self.maybe_button = QPushButton("Maybe Same Individual")
        self.diff_button = QPushButton("Different Individuals")
        control_layout.addWidget(self.same_button)
        control_layout.addWidget(self.maybe_button)
        control_layout.addWidget(self.diff_button)

        # Add user input field
        user_layout = QHBoxLayout()
        user_layout.addWidget(QLabel("User:"))
        self.user_field = QLineEdit()
        self.user_field.setPlaceholderText("Initials")
        self.user_field.setMaxLength(3)  # Limit to 3 characters for initials
        self.user_field.setFixedWidth(100)  # Set a fixed width to match button size
        user_layout.addWidget(self.user_field)
        control_layout.addLayout(user_layout)

        # Add notes field
        self.notes_field = QTextEdit()
        self.notes_field.setPlaceholderText("Enter notes about the comparison here...")
        control_layout.addWidget(QLabel("Notes:"))
        control_layout.addWidget(self.notes_field)

        main_layout.addLayout(control_layout)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals
        self.dir_combo_a.currentIndexChanged.connect(lambda: self.load_directory("A"))
        self.dir_combo_b.currentIndexChanged.connect(lambda: self.load_directory("B"))
        self.next_button_a.clicked.connect(lambda: self.next_image("A"))
        self.prev_button_a.clicked.connect(lambda: self.prev_image("A"))
        self.next_button_b.clicked.connect(lambda: self.next_image("B"))
        self.prev_button_b.clicked.connect(lambda: self.prev_image("B"))
        self.same_button.clicked.connect(lambda: self.compare_images('same'))
        self.diff_button.clicked.connect(lambda: self.compare_images('different'))
        self.maybe_button.clicked.connect(lambda: self.compare_images('maybe'))

        # Load initial images
        self.load_directory("A")
        self.load_directory("B")

    def load_directory(self, panel: str):
        index = self.dir_combo_a.currentIndex() if panel == "A" else self.dir_combo_b.currentIndex()
        directory = self.directories[index]
        self.current_images[panel] = [os.path.join(directory, f) for f in os.listdir(directory) if
                                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_indices[panel] = 0
        if self.current_images[panel]:
            self.load_image(panel)
        else:
            logging.warning(f"No images found in directory: {directory}")
            getattr(self, f"panel_{panel.lower()}").image_label.setText("No images found")

    def load_image(self, panel: str):
        if self.current_images[panel]:
            image_path = self.current_images[panel][self.current_indices[panel]]
            getattr(self, f"panel_{panel.lower()}").load_image(image_path)
        else:
            logging.warning(f"No images available for panel {panel}")

    def next_image(self, panel: str):
        self.change_image(panel, 1)

    def prev_image(self, panel: str):
        self.change_image(panel, -1)

    def change_image(self, panel: str, step: int):
        if self.current_images[panel]:
            self.current_indices[panel] = (self.current_indices[panel] + step) % len(self.current_images[panel])
            self.load_image(panel)

    def compare_images(self, status: str):
        try:
            logging.info(f"Images marked as {status}")

            id_a = self.codes[self.dir_combo_a.currentIndex()]
            id_b = self.codes[self.dir_combo_b.currentIndex()]
            full_path_a = self.current_images["A"][self.current_indices["A"]]
            full_path_b = self.current_images["B"][self.current_indices["B"]]

            notes = self.notes_field.toPlainText()
            user = self.user_field.text()

            if not user:
                QMessageBox.warning(self, "Missing User", "Please enter your initials before comparing images.")
                return

            self.comparison_logger.log_comparison(
                id_a, id_b, status, full_path_a, full_path_b,
                self.panel_a.rotation.value(), self.panel_a.brightness.value(), self.panel_a.contrast.value(),
                self.panel_a.color_correction_slider.value(),
                self.panel_b.rotation.value(), self.panel_b.brightness.value(), self.panel_b.contrast.value(),
                self.panel_b.color_correction_slider.value(),
                notes,
                user  # Add user to the log_comparison call
            )

        except Exception as e:
            print(f"Error in compare_images: {str(e)}")
            logging.error(f"Error in compare_images: {str(e)}", exc_info=True)

def parse_arguments() -> Dict[str, List[str]]:
    """
    Parse command-line arguments for directories and codes.
    """
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("Usage: python script.py [directory1] [code1] [directory2] [code2] ...")
        sys.exit(1)

    directories = sys.argv[1::2]
    codes = sys.argv[2::2]

    if len(directories) != len(codes):
        print("Error: Number of directories must match number of codes.")
        sys.exit(1)

    return {"directories": directories, "codes": codes}


def get_src_data_dir():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        return sys.argv[1]
    return os.path.join('..', 'archive', 'cropped_sequence_sorted')


def main(image_directories=None, codes=None):
    src_data_dir = get_src_data_dir()

    if image_directories is None:
        image_directories = glob.glob(os.path.join(src_data_dir, '*', '*'))
        image_directories = [d for d in image_directories if os.path.isdir(d)]

    if codes is None:
        codes = [os.path.basename(d) + '__' + os.path.basename(os.path.dirname(d)) for d in image_directories]

    for c, d in zip(codes, image_directories):
        print(f"Code: {c}, Directory: {d}")

    app = QApplication(sys.argv)
    window = MainWindow(image_directories, codes)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
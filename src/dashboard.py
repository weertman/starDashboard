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
from itertools import cycle
import glob
from typing import List, Dict
import logging
from datetime import datetime
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QPushButton, QComboBox, QFileDialog, QTextEdit,
                             QLineEdit, QMessageBox, QCheckBox)
from PyQt6.QtGui import QImage, QPixmap, QTransform, QPainter, QWheelEvent, QMouseEvent, QPen, QAction
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
                        'ZOOM_A', 'PAN_X_A', 'PAN_Y_A', 'ZOOM_B', 'PAN_X_B', 'PAN_Y_B',
                        'NOTES', 'USER']
        self.numeric_columns = ['ROTATION_A', 'BRIGHTNESS_A', 'CONTRAST_A', 'RED_SHIFT_A',
                                'ROTATION_B', 'BRIGHTNESS_B', 'CONTRAST_B', 'RED_SHIFT_B',
                                'ZOOM_A', 'PAN_X_A', 'PAN_Y_A', 'ZOOM_B', 'PAN_X_B', 'PAN_Y_B']
        self.string_columns = ['ID_A', 'ID_B', 'ARE_SAME', 'FULL_PATH_A', 'FULL_PATH_B', 'NOTES', 'USER']
        self.make_comparison_dir()
        self.create_csv_if_not_exists()

    def make_comparison_dir(self):
        base_dir = os.path.dirname(self.csv_path)
        comparison_dir = os.path.join(base_dir, 'pair_comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        for status in ['same', 'different', 'maybe']:
            status_dir = os.path.join(comparison_dir, status)
            os.makedirs(status_dir, exist_ok=True)

    def create_csv_if_not_exists(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.csv_path, index=False)
            logging.info(f"Created new comparison results CSV at {self.csv_path}")

    def log_comparison(self, id_a, id_b, are_same, full_path_a, full_path_b,
                       rotation_a, brightness_a, contrast_a, color_correction_a,
                       rotation_b, brightness_b, contrast_b, color_correction_b,
                       notes, user, view_settings_a, view_settings_b):
        try:
            logging.info(f"Logging comparison: {id_a} vs {id_b}, status={are_same}, user={user}")

            current_datetime = datetime.now()
            current_date = current_datetime.strftime('%d-%m-%Y')
            current_time = current_datetime.strftime('%H-%M-%S')

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
                'ZOOM_A': view_settings_a['zoom'],
                'PAN_X_A': view_settings_a['pan_x'],
                'PAN_Y_A': view_settings_a['pan_y'],
                'ZOOM_B': view_settings_b['zoom'],
                'PAN_X_B': view_settings_b['pan_x'],
                'PAN_Y_B': view_settings_b['pan_y'],
                'NOTES': notes,
                'USER': user
            }

            # Ensure numeric columns are float
            for col in self.numeric_columns:
                new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

            # Ensure string columns are str
            for col in self.string_columns:
                new_data[col] = str(new_data[col])

            try:
                df = pd.read_csv(self.csv_path)
                for col in self.numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                for col in self.string_columns:
                    df[col] = df[col].astype(str)
            except FileNotFoundError:
                df = pd.DataFrame(columns=self.columns)

            logging.debug(f"Selecting row with ID_A={id_a}, ID_B={id_b}, USER={user}")
            mask = (df['ID_A'] == id_a) & (df['ID_B'] == id_b) & (df['USER'] == user)
            logging.debug(f"Number of rows selected: {mask.sum()}")

            if mask.sum() == 0:
                # No matching row, append new row
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            elif mask.sum() == 1:
                # One matching row, update it
                for column, value in new_data.items():
                    if column in self.numeric_columns:
                        df.loc[mask, column] = pd.to_numeric(value, errors='coerce')
                    else:
                        df.loc[mask, column] = value
            else:
                # Multiple matching rows, this shouldn't happen
                raise ValueError(
                    f"Multiple rows ({mask.sum()}) found for ID_A={id_a}, ID_B={id_b}, USER={user}. This indicates a data integrity issue.")

            # Ensure all columns are present
            for col in self.columns:
                if col not in df.columns:
                    df[col] = np.nan

            # Reorder columns to match self.columns
            df = df[self.columns]

            df.to_csv(self.csv_path, index=False)
            logging.info(f"Comparison logged successfully")
        except Exception as e:
            logging.error(f"Error logging comparison: {str(e)}", exc_info=True)
            raise  # Re-raise the exception after logging

    def get_last_logged_comparison(self, id_a, id_b):
        try:
            # Load the CSV file
            df = pd.read_csv(self.csv_path)

            # Filter for entries matching either ID_A vs ID_B or ID_B vs ID_A
            mask = ((df['ID_A'] == id_a) & (df['ID_B'] == id_b)) | ((df['ID_A'] == id_b) & (df['ID_B'] == id_a))

            # Apply the mask to get the relevant entries
            relevant_entries = df[mask]

            if relevant_entries.empty:
                # If no entries are found, return None or appropriate default value
                return None

            # Convert DATE and TIME columns to a datetime object for sorting
            relevant_entries['DATETIME'] = pd.to_datetime(relevant_entries['DATE'] + ' ' + relevant_entries['TIME'],
                                                          format='%d-%m-%Y %H-%M-%S')

            # Sort by the DATETIME column in ascending order
            relevant_entries = relevant_entries.sort_values(by='DATETIME')

            # The last entry is the most recent
            last_entry = relevant_entries.iloc[-1]

            # Return the last entry as a dictionary or a pandas Series
            return last_entry.to_dict()

        except FileNotFoundError:
            # Handle the case where the CSV file does not exist
            logging.error("Comparison log file not found.")
            return None

        except Exception as e:
            # General exception handling
            logging.error(f"Error retrieving last logged comparison: {str(e)}", exc_info=True)
            return None

    def get_all_comparisons_with_settings(self, id_a, id_b):
        # Similar to get_all_comparisons, but include image settings
        try:
            df = pd.read_csv(self.csv_path)
            mask = ((df['ID_A'] == id_a) & (df['ID_B'] == id_b)) | ((df['ID_A'] == id_b) & (df['ID_B'] == id_a))
            relevant_entries = df[mask].copy()

            if relevant_entries.empty:
                return []

            relevant_entries['DATETIME'] = pd.to_datetime(relevant_entries['DATE'] + ' ' + relevant_entries['TIME'],
                                                          format='%d-%m-%Y %H-%M-%S')
            relevant_entries = relevant_entries.sort_values(by='DATETIME', ascending=False)

            return [(row['ARE_SAME'], row['USER'], row['DATETIME'].strftime('%Y-%m-%d %H:%M:%S'),
                     {'rotation_a': row['ROTATION_A'], 'brightness_a': row['BRIGHTNESS_A'],
                      'contrast_a': row['CONTRAST_A'], 'red_shift_a': row['RED_SHIFT_A'],
                      'zoom_a': row['ZOOM_A'], 'pan_x_a': row['PAN_X_A'], 'pan_y_a': row['PAN_Y_A'],
                      'rotation_b': row['ROTATION_B'], 'brightness_b': row['BRIGHTNESS_B'],
                      'contrast_b': row['CONTRAST_B'], 'red_shift_b': row['RED_SHIFT_B'],
                      'zoom_b': row['ZOOM_B'], 'pan_x_b': row['PAN_X_B'], 'pan_y_b': row['PAN_Y_B']})
                    for _, row in relevant_entries.iterrows()]
        except Exception as e:
            logging.error(f"Error retrieving comparisons with settings: {str(e)}", exc_info=True)
            return []

    def get_all_comparisons(self, id_a, id_b):
        try:
            df = pd.read_csv(self.csv_path)
            mask = ((df['ID_A'] == id_a) & (df['ID_B'] == id_b)) | ((df['ID_A'] == id_b) & (df['ID_B'] == id_a))
            relevant_entries = df[mask].copy()  # Create an explicit copy

            if relevant_entries.empty:
                return []

            relevant_entries['DATETIME'] = pd.to_datetime(relevant_entries['DATE'] + ' ' + relevant_entries['TIME'],
                                                          format='%d-%m-%Y %H-%M-%S')
            relevant_entries = relevant_entries.sort_values(by='DATETIME', ascending=False)

            return [(row['ARE_SAME'], row['USER'], row['DATETIME'].strftime('%Y-%m-%d %H:%M:%S'))
                    for _, row in relevant_entries.iterrows()]
        except Exception as e:
            logging.error(f"Error retrieving comparisons: {str(e)}", exc_info=True)
            return []

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
        self.show_crosshair = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Image container (no changes here)
        self.image_container = QWidget()
        self.image_container.setFixedSize(800, 800)
        self.image_container_layout = QVBoxLayout(self.image_container)
        self.image_container_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_container_layout.addWidget(self.image_label)

        layout.addWidget(self.image_container)

        # Sliders (updated here)
        for name, label_text in [("brightness", "Brightness:"),
                                 ("contrast", "Contrast:"),
                                 ("rotation", "Rotation:"),
                                 ("color_correction_slider", "Red Shift:")]:
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(label_text))
            slider = QSlider(Qt.Orientation.Horizontal)
            setattr(self, name, slider)
            slider_layout.addWidget(slider)
            layout.addLayout(slider_layout)

        # Set slider ranges and default values
        self.brightness.setRange(-100, 100)
        self.contrast.setRange(0, 200)
        self.rotation.setRange(-180, 180)
        self.color_correction_slider.setRange(-100, 100)

        self.brightness.setValue(0)
        self.contrast.setValue(100)
        self.rotation.setValue(0)
        self.color_correction_slider.setValue(0)

        # Reset button
        self.reset_button = QPushButton("Reset Image")
        layout.addWidget(self.reset_button)

        # Crosshair toggle
        self.crosshair_toggle = QCheckBox("Show Crosshair")
        layout.addWidget(self.crosshair_toggle)

        self.setLayout(layout)

        # Connect signals
        self.brightness.valueChanged.connect(self.update_image)
        self.contrast.valueChanged.connect(self.update_image)
        self.rotation.valueChanged.connect(self.update_image)
        self.color_correction_slider.valueChanged.connect(self.update_image)
        self.reset_button.clicked.connect(self.reset_image)
        self.crosshair_toggle.stateChanged.connect(self.toggle_crosshair)

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
        self.update_crosshair()

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
            self.color_correction_slider.setValue(0)
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

    def toggle_crosshair(self, state):
        print(f'Crosshair toggle called with state: {state}')
        self.show_crosshair = (state == Qt.CheckState.Checked.value)
        print(f'Crosshair toggled: {self.show_crosshair}')
        self.update_image()

    def update_crosshair(self):
        if self.image_label.pixmap():
            pixmap = self.image_label.pixmap().copy()
            if self.show_crosshair:
                painter = QPainter(pixmap)
                painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))

                # Define hatch properties
                hatch_length = 10  # Length of hatches in pixels
                hatch_spacing = 40  # Spacing between hatches in pixels

                # Draw vertical line
                center_x = pixmap.width() // 2
                center_y = pixmap.height() // 2
                painter.drawLine(center_x, 0, center_x, pixmap.height())

                # Draw horizontal line
                painter.drawLine(0, center_y, pixmap.width(), center_y)

                # Add hatches along the vertical line
                for i in range(0, pixmap.height(), hatch_spacing):
                    painter.drawLine(center_x - hatch_length // 2, i, center_x + hatch_length // 2, i)

                # Add hatches along the horizontal line
                for i in range(0, pixmap.width(), hatch_spacing):
                    painter.drawLine(i, center_y - hatch_length // 2, i, center_y + hatch_length // 2)

                painter.end()

            self.image_label.setPixmap(pixmap)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.show_crosshair:
            painter = QPainter(self.image_container)
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))

            # Define hatch properties
            hatch_length = 10  # Length of hatches in pixels
            hatch_spacing = 40  # Spacing between hatches in pixels

            # Draw vertical line with hatches
            center_x = self.image_container.width() // 2
            center_y = self.image_container.height() // 2
            painter.drawLine(center_x, 0, center_x, self.image_container.height())

            for i in range(0, self.image_container.height(), hatch_spacing):
                painter.drawLine(center_x - hatch_length // 2, i, center_x + hatch_length // 2, i)

            # Draw horizontal line with hatches
            painter.drawLine(0, center_y, self.image_container.width(), center_y)

            for i in range(0, self.image_container.width(), hatch_spacing):
                painter.drawLine(i, center_y - hatch_length // 2, i, center_y + hatch_length // 2)

            painter.end()

    def get_view_settings(self):
        return {
            'zoom': self.zoom_factor,
            'pan_x': self.pan_offset.x(),
            'pan_y': self.pan_offset.y()
        }

    def save_current_view(self, save_path):
        if self.image_label.pixmap():
            ## if crosshair is shown, turn it off then save
            crosshair_state = self.show_crosshair
            if crosshair_state:
                self.show_crosshair = False
                self.update_image()

            pixmap = self.image_label.pixmap()
            pixmap.save(save_path, 'PNG')
            logging.info(f"Saved current view to {save_path}")

            ## turn crosshair back on if it was on
            if crosshair_state:
                self.show_crosshair = True
                self.update_image()

        else:
            logging.warning("No image to save")

    def apply_settings(self, settings):
        self.zoom_factor = settings['zoom']
        self.pan_offset = QPointF(settings['pan_x'], settings['pan_y'])
        self.rotation.setValue(settings['rotation'])
        self.brightness.setValue(settings['brightness'])
        self.contrast.setValue(settings['contrast'])
        self.color_correction_slider.setValue(settings['red_shift'])
        self.update_image()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.directories = []
        self.codes = []
        self.comparison_logger = None  # Initialize with None
        self.current_images = {"A": [], "B": []}
        self.current_indices = {"A": 0, "B": 0}
        self.init_ui()
        self.init_menu()

    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        select_action = QAction('Select root folder', self)
        select_action.triggered.connect(self.select_root_folder)
        file_menu.addAction(select_action)

    def disable_ui_elements(self):
        self.dir_combo_a.setEnabled(False)
        self.dir_combo_b.setEnabled(False)
        self.prev_button_a.setEnabled(False)
        self.next_button_a.setEnabled(False)
        self.prev_button_b.setEnabled(False)
        self.next_button_b.setEnabled(False)
        self.same_button.setEnabled(False)
        self.maybe_button.setEnabled(False)
        self.diff_button.setEnabled(False)

    def enable_ui_elements(self):
        self.dir_combo_a.setEnabled(True)
        self.dir_combo_b.setEnabled(True)
        self.prev_button_a.setEnabled(True)
        self.next_button_a.setEnabled(True)
        self.prev_button_b.setEnabled(True)
        self.next_button_b.setEnabled(True)
        self.same_button.setEnabled(True)
        self.maybe_button.setEnabled(True)
        self.diff_button.setEnabled(True)

    def select_root_folder(self):
        try:
            root_dir = QFileDialog.getExistingDirectory(self, "Select Root Folder")
            if root_dir:
                logging.info(f"Selected root directory: {root_dir}")
                self.load_directories(root_dir)
        except Exception as e:
            logging.error(f"Error in select_root_folder: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred while selecting the root folder: {str(e)}")

    def load_directories(self, root_dir):
        try:
            logging.info("Starting to load directories")
            image_directories = glob.glob(os.path.join(root_dir, '*', '*'))
            image_directories = [d for d in image_directories if os.path.isdir(d) and 'pair_comparisons' not in d]

            if not image_directories:
                raise ValueError("No valid image directories found")

            self.directories = image_directories
            self.codes = [os.path.basename(d) + '__' + os.path.basename(os.path.dirname(d)) for d in image_directories]

            logging.info(f"Found {len(self.directories)} directories")

            # Initialize ComparisonLogger before updating directory combos
            self.comparison_logger = ComparisonLogger(root_dir)
            logging.info("ComparisonLogger initialized")

            self.update_directory_combos()
            self.enable_ui_elements()
            logging.info("Directories loaded successfully")
        except Exception as e:
            logging.error(f"Error in load_directories: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred while loading directories: {str(e)}")

    def update_directory_combos(self):
        try:
            logging.info("Updating directory combos")
            self.dir_combo_a.clear()
            self.dir_combo_b.clear()
            self.dir_combo_a.addItems(self.codes)
            self.dir_combo_b.addItems(self.codes)
            if self.directories:
                self.load_directory("A")
                self.load_directory("B")
            logging.info("Directory combos updated successfully")
        except Exception as e:
            logging.error(f"Error in update_directory_combos: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred while updating directory combos: {str(e)}")

    def load_directory(self, panel: str):
        try:
            if self.directories:
                logging.info(f"Loading directory for panel {panel}")
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
                logging.info(f"Directory loaded for panel {panel}")
        except Exception as e:
            logging.error(f"Error in load_directory: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred while loading directory for panel {panel}: {str(e)}")

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel
        left_panel = QVBoxLayout()
        self.panel_a = ImagePanel(self)
        self.dir_combo_a = QComboBox()
        self.dir_combo_a.addItems(self.codes)
        left_panel.addWidget(self.dir_combo_a)
        left_panel.addWidget(self.panel_a)

        # Navigation buttons A
        nav_buttons_layout_a = QHBoxLayout()
        self.prev_button_a = QPushButton("Previous Image A")
        self.next_button_a = QPushButton("Next Image A")
        nav_buttons_layout_a.addWidget(self.prev_button_a)
        nav_buttons_layout_a.addWidget(self.next_button_a)
        left_panel.addLayout(nav_buttons_layout_a)

        main_layout.addLayout(left_panel)

        # Right panel
        right_panel = QVBoxLayout()
        self.panel_b = ImagePanel(self)
        self.dir_combo_b = QComboBox()
        self.dir_combo_b.addItems(self.codes)
        right_panel.addWidget(self.dir_combo_b)
        right_panel.addWidget(self.panel_b)

        # Navigation buttons B
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

        # User input field
        user_layout = QHBoxLayout()
        user_layout.addWidget(QLabel("User:"))
        self.user_field = QLineEdit()
        self.user_field.setPlaceholderText("Initials")
        self.user_field.setMaxLength(10)
        self.user_field.setFixedWidth(100)
        user_layout.addWidget(self.user_field)
        control_layout.addLayout(user_layout)

        # Notes field
        self.notes_field = QTextEdit()
        self.notes_field.setPlaceholderText("Enter notes about the comparison here...")
        control_layout.addWidget(QLabel("Notes:"))
        control_layout.addWidget(self.notes_field)

        # Add new widget for past matches
        self.past_matches_field = QTextEdit()
        self.past_matches_field.setReadOnly(True)
        self.past_matches_field.setPlaceholderText("No past matches found...")
        control_layout.addWidget(QLabel("Past Matches:"))
        control_layout.addWidget(self.past_matches_field)

        # Add crosshair toggles to the control layout
        control_layout.addWidget(QLabel("Crosshair Controls:"))
        control_layout.addWidget(self.panel_a.crosshair_toggle)
        control_layout.addWidget(self.panel_b.crosshair_toggle)

        self.cycle_perspectives_button = QPushButton("Cycle Past Perspectives")
        control_layout.addWidget(self.cycle_perspectives_button)

        # Add the new Reset Perspective button
        self.reset_perspective_button = QPushButton("Reset Perspective")
        control_layout.addWidget(self.reset_perspective_button)

        self.past_perspectives = []
        self.perspective_cycle = None

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
        self.cycle_perspectives_button.clicked.connect(self.cycle_past_perspectives)
        self.reset_perspective_button.clicked.connect(self.reset_perspective)

        # Disable UI elements initially
        self.disable_ui_elements()

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
        try:
            if self.current_images[panel]:
                logging.info(f"Loading image for panel {panel}")
                image_path = self.current_images[panel][self.current_indices[panel]]
                getattr(self, f"panel_{panel.lower()}").load_image(image_path)
                if self.comparison_logger:  # Only update past matches if comparison_logger is initialized
                    self.update_past_matches()
                logging.info(f"Image loaded for panel {panel}")
            else:
                logging.warning(f"No images available for panel {panel}")
        except Exception as e:
            logging.error(f"Error in load_image: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred while loading image for panel {panel}: {str(e)}")

    def update_past_matches(self):
        try:
            if self.comparison_logger is None:
                logging.warning("ComparisonLogger is not initialized")
                return

            id_a = self.codes[self.dir_combo_a.currentIndex()]
            id_b = self.codes[self.dir_combo_b.currentIndex()]
            self.past_perspectives = self.comparison_logger.get_all_comparisons_with_settings(id_a, id_b)

            if self.past_perspectives:
                text = "\n".join(f"({match}, {user}, {datetime})" for match, user, datetime, _ in self.past_perspectives)
                self.perspective_cycle = cycle(self.past_perspectives)
            else:
                text = "No past matches found."
                self.perspective_cycle = None

            self.past_matches_field.setText(text)
        except Exception as e:
            logging.error(f"Error in update_past_matches: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "Warning", f"An error occurred while updating past matches: {str(e)}")

    def cycle_past_perspectives(self):
        if not self.perspective_cycle:
            QMessageBox.information(self, "No Past Perspectives", "There are no past perspectives to cycle through.")
            return

        match, user, datetime, settings = next(self.perspective_cycle)
        self.panel_a.apply_settings({k[:-2]: v for k, v in settings.items() if k.endswith('_a')})
        self.panel_b.apply_settings({k[:-2]: v for k, v in settings.items() if k.endswith('_b')})

        QMessageBox.information(self, "Past Perspective Applied", f"Applied perspective from {user} on {datetime}")

    def reset_perspective(self):
        # Reset both image panels
        self.panel_a.reset_image()
        self.panel_b.reset_image()

        # Reset any MainWindow state related to perspectives
        self.perspective_cycle = None

        # Optionally, update the UI to reflect the reset state
        self.update_past_matches()

    def next_image(self, panel: str):
        self.change_image(panel, 1)

    def prev_image(self, panel: str):
        self.change_image(panel, -1)

    def change_image(self, panel: str, step: int):
        if self.current_images[panel]:
            self.current_indices[panel] = (self.current_indices[panel] + step) % len(self.current_images[panel])
            self.load_image(panel)

    def save_comparison_images(self, status, id_a, id_b):
        base_dir = os.path.dirname(self.comparison_logger.csv_path)
        comparison_dir = os.path.join(base_dir, 'pair_comparisons', status, f'{id_a}_vs_{id_b}')
        os.makedirs(comparison_dir, exist_ok=True)

        image_a_path = os.path.join(comparison_dir, f'A__{id_a}.png')
        image_b_path = os.path.join(comparison_dir, f'B__{id_b}.png')

        self.panel_a.save_current_view(image_a_path)
        self.panel_b.save_current_view(image_b_path)

        logging.info(f"Saved comparison images to {comparison_dir}")

    def compare_images(self, status: str):
        try:
            logging.info(f"Images marked as {status}")

            id_a = self.codes[self.dir_combo_a.currentIndex()]
            id_b = self.codes[self.dir_combo_b.currentIndex()]
            full_path_a = self.current_images["A"][self.current_indices["A"]]
            full_path_b = self.current_images["B"][self.current_indices["B"]]

            notes = self.notes_field.toPlainText()
            user = self.user_field.text()

            view_settings_a = self.panel_a.get_view_settings()
            view_settings_b = self.panel_b.get_view_settings()

            if not user:
                QMessageBox.warning(self, "Missing User", "Please enter your initials before comparing images.")
                return

            self.comparison_logger.log_comparison(
                id_a, id_b, status, full_path_a, full_path_b,
                self.panel_a.rotation.value(), self.panel_a.brightness.value(), self.panel_a.contrast.value(),
                self.panel_a.color_correction_slider.value(),
                # Changed from color_correction to color_correction_slider
                self.panel_b.rotation.value(), self.panel_b.brightness.value(), self.panel_b.contrast.value(),
                self.panel_b.color_correction_slider.value(),
                # Changed from color_correction to color_correction_slider
                notes,
                user,
                view_settings_a,
                view_settings_b
            )
            self.update_past_matches()  # Update past matches after new comparison

            # Save comparison images
            self.save_comparison_images(status, id_a, id_b)

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


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
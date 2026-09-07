#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Folder inference dialog for YOLO detection and OBB models."""

import csv
import math
import os
from datetime import datetime

try:
    from PyQt5.QtCore import Qt, QSettings, QStandardPaths, QThread, QObject, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
        QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
        QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt4.QtCore import Qt, QSettings, QThread, QObject, pyqtSignal, pyqtSlot
    from PyQt4.QtGui import (
        QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
        QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QImage, QLineEdit, QMessageBox, QPixmap, QProgressBar, QPushButton,
        QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    )
    QStandardPaths = None


IMAGE_EXTENSIONS = ('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp')
MODEL_EXTENSIONS = ('.onnx', '.pt')


def inference_settings_path():
    """Return the explicit, user-local INI path used by this dialog."""
    if QStandardPaths is not None:
        config_dir = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.GenericConfigLocation),
            'roLabelImg')
    else:
        config_dir = os.path.join(os.path.expanduser('~'), '.roLabelImg')
    return os.path.join(config_dir, 'inference.ini')


def discover_models(project_dir):
    candidates = []
    roots = [
        os.path.join(project_dir, 'training', 'yolo11_obb', 'export'),
        os.path.join(project_dir, 'training', 'yolo11_obb', 'runs'),
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for current_root, _dirs, files in os.walk(root):
            for filename in files:
                path = os.path.abspath(os.path.join(current_root, filename))
                if filename.lower().endswith(MODEL_EXTENSIONS):
                    candidates.append(path)
    return sorted(set(candidates), key=lambda value: value.lower())


def collect_images(folder_path, include_subfolders):
    images = []
    if not os.path.isdir(folder_path):
        return images
    if include_subfolders:
        iterator = os.walk(folder_path)
    else:
        try:
            iterator = [(folder_path, [], os.listdir(folder_path))]
        except OSError:
            return images
    for root, _dirs, files in iterator:
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.abspath(os.path.join(root, filename)))
    return sorted(images, key=lambda value: value.lower())


def _safe_output_path(output_dir, source_root, image_path):
    relative_path = os.path.relpath(image_path, source_root)
    relative_root, _extension = os.path.splitext(relative_path)
    return os.path.join(output_dir, relative_root + '.png')


def _write_image(path, image):
    """Write an OpenCV image while supporting non-ASCII Windows paths."""
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise IOError('Could not encode prediction image.')
    encoded.tofile(path)


def _read_image(path):
    """Read an OpenCV image while supporting non-ASCII Windows paths."""
    import cv2
    import numpy as np
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError('Could not decode image: %s' % path)
    return image


def extract_detections(result):
    """Convert Ultralytics tensor results into serializable detection dicts."""
    detections = []
    names = result.names
    if result.obb is not None and result.obb.conf is not None:
        confidences = result.obb.conf.detach().cpu().tolist()
        class_ids = result.obb.cls.detach().cpu().tolist()
        boxes = result.obb.xywhr.detach().cpu().tolist()
        corners = result.obb.xyxyxyxy.detach().cpu().tolist()
        for confidence, class_id, box, points in zip(
                confidences, class_ids, boxes, corners):
            class_id = int(class_id)
            detections.append({
                'kind': 'obb',
                'class_id': class_id,
                'class_name': str(names[class_id]),
                'score': float(confidence),
                'cx': float(box[0]),
                'cy': float(box[1]),
                'width': float(box[2]),
                'height': float(box[3]),
                'angle': float(box[4]) % math.pi,
                'points': [[float(point[0]), float(point[1])]
                           for point in points],
            })
    elif result.boxes is not None and result.boxes.conf is not None:
        confidences = result.boxes.conf.detach().cpu().tolist()
        class_ids = result.boxes.cls.detach().cpu().tolist()
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        for confidence, class_id, box in zip(confidences, class_ids, boxes):
            class_id = int(class_id)
            detections.append({
                'kind': 'box',
                'class_id': class_id,
                'class_name': str(names[class_id]),
                'score': float(confidence),
                'xmin': float(box[0]),
                'ymin': float(box[1]),
                'xmax': float(box[2]),
                'ymax': float(box[3]),
            })
    return detections


def filter_detections(detections, class_name, minimum_score):
    return [
        detection for detection in detections
        if detection['score'] >= minimum_score
        and (not class_name or detection['class_name'] == class_name)
    ]


def render_detections(image, detections):
    """Render filtered detections and a width-axis arrow for every YOLO OBB."""
    import cv2
    import numpy as np

    rendered = image.copy()
    for detection in detections:
        if detection['kind'] == 'obb':
            points = np.asarray(detection['points'], dtype=np.int32)
            cv2.polylines(rendered, [points], True, (255, 80, 0), 3,
                          lineType=cv2.LINE_AA)
            cx = detection['cx']
            cy = detection['cy']
            arrow_length = max(12.0, detection['width'] * 0.35)
            end = (
                int(round(cx + math.cos(detection['angle']) * arrow_length)),
                int(round(cy + math.sin(detection['angle']) * arrow_length)),
            )
            start = (int(round(cx)), int(round(cy)))
            cv2.arrowedLine(rendered, start, end, (255, 255, 255), 8,
                            line_type=cv2.LINE_AA, tipLength=0.30)
            cv2.arrowedLine(rendered, start, end, (0, 0, 255), 5,
                            line_type=cv2.LINE_AA, tipLength=0.30)
            label_anchor = tuple(points[2])
        else:
            start = (int(round(detection['xmin'])), int(round(detection['ymin'])))
            end = (int(round(detection['xmax'])), int(round(detection['ymax'])))
            cv2.rectangle(rendered, start, end, (255, 80, 0), 3,
                          lineType=cv2.LINE_AA)
            label_anchor = end

        label = '%s %.2f' % (detection['class_name'], detection['score'])
        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = max(0, min(int(label_anchor[0]), rendered.shape[1] - text_size[0] - 4))
        y = max(text_size[1] + 4, min(int(label_anchor[1]), rendered.shape[0] - baseline))
        cv2.rectangle(rendered, (x, y - text_size[1] - 4),
                      (x + text_size[0] + 4, y + baseline), (255, 80, 0), -1)
        cv2.putText(rendered, label, (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)
    return rendered


def cv_image_to_qpixmap(image):
    import cv2
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, channels * width,
                    QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class InferenceWorker(QObject):
    progress = pyqtSignal(object)
    failed = pyqtSignal(str)
    completed = pyqtSignal(object)

    def __init__(self, model_path, source_folder, image_paths, output_dir,
                 confidence, iou_threshold):
        super(InferenceWorker, self).__init__()
        self.model_path = model_path
        self.source_folder = source_folder
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.cancel_requested = False

    @pyqtSlot()
    def cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        rows = []
        try:
            from ultralytics import YOLO

            os.makedirs(self.output_dir, exist_ok=True)
            model = YOLO(self.model_path)
            for index, image_path in enumerate(self.image_paths, 1):
                if self.cancel_requested:
                    break

                started = datetime.now()
                try:
                    output_path = _safe_output_path(
                        self.output_dir, self.source_folder, image_path)
                    result = model.predict(
                        source=image_path,
                        conf=self.confidence,
                        iou=self.iou_threshold,
                        verbose=False,
                    )[0]
                    detection_items = extract_detections(result)
                    source_image = _read_image(image_path)
                    _write_image(
                        output_path,
                        render_detections(source_image, detection_items))
                    detections = len(detection_items)
                    max_confidence = max(
                        (item['score'] for item in detection_items), default=None)
                    image_shape = list(source_image.shape)
                    status = 'Completed'
                    error_message = ''
                except Exception as image_error:
                    output_path = ''
                    detection_items = []
                    detections = None
                    max_confidence = None
                    image_shape = None
                    status = 'Failed'
                    error_message = '%s: %s' % (
                        type(image_error).__name__, image_error)

                elapsed_ms = (datetime.now() - started).total_seconds() * 1000.0
                row = {
                    'index': index,
                    'image_path': image_path,
                    'relative_path': os.path.relpath(image_path, self.source_folder),
                    'output_path': output_path,
                    'detections': detections,
                    'detection_items': detection_items,
                    'max_confidence': max_confidence,
                    'image_shape': image_shape,
                    'elapsed_ms': elapsed_ms,
                    'status': status,
                    'error': error_message,
                }
                rows.append(row)
                self.progress.emit(row)

            summary_path = os.path.join(self.output_dir, 'prediction_summary.csv')
            with open(summary_path, 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    'image', 'detections', 'max_confidence', 'elapsed_ms',
                    'status', 'error', 'prediction_image'))
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        'image': row['relative_path'],
                        'detections': row['detections'],
                        'max_confidence': '' if row['max_confidence'] is None
                        else '%.6f' % row['max_confidence'],
                        'elapsed_ms': '%.1f' % row['elapsed_ms'],
                        'status': row['status'],
                        'error': row['error'],
                        'prediction_image': '' if not row['output_path']
                        else os.path.relpath(row['output_path'], self.output_dir),
                    })

            self.completed.emit({
                'cancelled': self.cancel_requested,
                'processed': len(rows),
                'failed': sum(row['status'] == 'Failed' for row in rows),
                'total': len(self.image_paths),
                'output_dir': self.output_dir,
                'summary_path': summary_path,
            })
        except Exception as error:
            self.failed.emit('%s: %s' % (type(error).__name__, error))


class InferenceFolderDialog(QDialog):
    cancel_worker = pyqtSignal()

    def __init__(self, project_dir, parent=None):
        super(InferenceFolderDialog, self).__init__(parent)
        self.project_dir = os.path.abspath(project_dir)
        self.settings_path = inference_settings_path()
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        self.settings = QSettings(self.settings_path, QSettings.IniFormat)
        self.thread = None
        self.worker = None
        self.preview_pixmap = None
        self.results = []
        self.last_output_dir = None

        self.setWindowTitle('Inference Folder')
        self.resize(1180, 760)
        self.setMinimumSize(900, 600)
        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        settings_group = QGroupBox('Inference settings')
        settings_layout = QGridLayout(settings_group)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(discover_models(self.project_dir))
        self.model_button = QPushButton('Browse...')
        self.model_button.clicked.connect(self._browse_model)

        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_button = QPushButton('Browse...')
        self.folder_button.clicked.connect(self._browse_folder)

        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_button = QPushButton('Browse...')
        self.output_button.clicked.connect(self._browse_output)

        self.include_subfolders = QCheckBox('Include subfolders')
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)

        settings_layout.addWidget(QLabel('Model'), 0, 0)
        settings_layout.addWidget(self.model_combo, 0, 1)
        settings_layout.addWidget(self.model_button, 0, 2)
        settings_layout.addWidget(QLabel('Image folder'), 1, 0)
        settings_layout.addWidget(self.folder_edit, 1, 1)
        settings_layout.addWidget(self.folder_button, 1, 2)
        settings_layout.addWidget(QLabel('Output root'), 2, 0)
        settings_layout.addWidget(self.output_edit, 2, 1)
        settings_layout.addWidget(self.output_button, 2, 2)

        parameter_widget = QWidget()
        parameter_layout = QHBoxLayout(parameter_widget)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.addWidget(self.include_subfolders)
        parameter_layout.addSpacing(20)
        parameter_layout.addWidget(QLabel('Confidence'))
        parameter_layout.addWidget(self.confidence_spin)
        parameter_layout.addSpacing(12)
        parameter_layout.addWidget(QLabel('IoU'))
        parameter_layout.addWidget(self.iou_spin)
        parameter_layout.addStretch(1)
        settings_layout.addWidget(parameter_widget, 3, 1, 1, 2)

        self.class_filter = QComboBox()
        self.class_filter.addItem('All classes', '')
        self.score_filter = QDoubleSpinBox()
        self.score_filter.setRange(0.0, 1.0)
        self.score_filter.setSingleStep(0.05)
        self.score_filter.setDecimals(2)
        self.filter_summary = QLabel('No inference results')
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QLabel('Class name'))
        filter_layout.addWidget(self.class_filter)
        filter_layout.addSpacing(12)
        filter_layout.addWidget(QLabel('Minimum score'))
        filter_layout.addWidget(self.score_filter)
        filter_layout.addSpacing(12)
        filter_layout.addWidget(self.filter_summary, 1)
        settings_layout.addWidget(QLabel('Result filter'), 4, 0)
        settings_layout.addWidget(filter_widget, 4, 1, 1, 2)
        settings_layout.setColumnStretch(1, 1)

        self.class_filter.currentIndexChanged.connect(self._apply_filter)
        self.score_filter.valueChanged.connect(self._apply_filter)

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ('#', 'Image', 'Detections', 'Max confidence', 'Elapsed', 'Status'))
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setStretchLastSection(False)
        self.results_table.setColumnWidth(0, 45)
        self.results_table.setColumnWidth(1, 360)
        self.results_table.setColumnWidth(2, 90)
        self.results_table.setColumnWidth(3, 115)
        self.results_table.setColumnWidth(4, 85)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.itemSelectionChanged.connect(self._show_selected_result)

        self.preview_label = QLabel('Select a completed row to preview the prediction.')
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 320)
        self.preview_label.setStyleSheet('QLabel { background: #202020; color: #d0d0d0; }')

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.results_table)
        splitter.addWidget(self.preview_label)
        splitter.setSizes((620, 500))

        self.progress_bar = QProgressBar()
        self.status_label = QLabel('Ready')
        self.start_button = QPushButton('Start inference')
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton('Export filtered annotations')
        self.export_button.setEnabled(False)
        self.close_button = QPushButton('Close')
        self.start_button.clicked.connect(self._start_inference)
        self.cancel_button.clicked.connect(self._cancel_inference)
        self.export_button.clicked.connect(self._export_filtered_annotations)
        self.close_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.status_label, 1)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(settings_group)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.progress_bar)
        layout.addLayout(button_layout)

    def _load_settings(self):
        previous_model = str(self.settings.value('model_path', '') or '')
        if previous_model and self.model_combo.findText(previous_model) < 0:
            self.model_combo.addItem(previous_model)
        if previous_model:
            self.model_combo.setCurrentText(previous_model)
        elif self.model_combo.count():
            preferred = next((index for index in range(self.model_combo.count())
                              if self.model_combo.itemText(index).lower().endswith('.pt')), 0)
            self.model_combo.setCurrentIndex(preferred)

        self.folder_edit.setText(str(self.settings.value('source_folder', '') or ''))
        default_output = os.path.join(self.project_dir, 'prediction_results')
        self.output_edit.setText(str(
            self.settings.value('output_root', default_output) or default_output))
        self.include_subfolders.setChecked(
            str(self.settings.value('include_subfolders', 'true')).lower() == 'true')
        self.confidence_spin.setValue(float(self.settings.value('confidence', 0.25)))
        self.iou_spin.setValue(float(self.settings.value('iou_threshold', 0.70)))
        saved_class = str(self.settings.value('filter_class_name', '') or '')
        if saved_class:
            self.class_filter.addItem(saved_class, saved_class)
            self.class_filter.setCurrentIndex(self.class_filter.count() - 1)
        self.score_filter.setValue(float(self.settings.value('filter_score', 0.25)))

    def _save_settings(self):
        self.settings.setValue('model_path', self.model_combo.currentText().strip())
        self.settings.setValue('source_folder', self.folder_edit.text().strip())
        self.settings.setValue('output_root', self.output_edit.text().strip())
        self.settings.setValue('include_subfolders', self.include_subfolders.isChecked())
        self.settings.setValue('confidence', self.confidence_spin.value())
        self.settings.setValue('iou_threshold', self.iou_spin.value())
        self.settings.setValue('filter_class_name', self._selected_class_name())
        self.settings.setValue('filter_score', self.score_filter.value())
        self.settings.sync()

    def _browse_model(self):
        current = self.model_combo.currentText().strip() or self.project_dir
        path, _filter = QFileDialog.getOpenFileName(
            self, 'Select inference model', current,
            'YOLO models (*.pt *.onnx);;All files (*.*)')
        if path:
            if self.model_combo.findText(path) < 0:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)
            self._save_settings()

    def _browse_folder(self):
        current = self.folder_edit.text().strip() or self.project_dir
        path = QFileDialog.getExistingDirectory(self, 'Select image folder', current)
        if path:
            self.folder_edit.setText(path)
            self._save_settings()

    def _browse_output(self):
        current = self.output_edit.text().strip() or self.project_dir
        path = QFileDialog.getExistingDirectory(self, 'Select output root', current)
        if path:
            self.output_edit.setText(path)
            self._save_settings()

    def _validate(self):
        model_path = self.model_combo.currentText().strip()
        source_folder = self.folder_edit.text().strip()
        output_root = self.output_edit.text().strip()
        if not os.path.isfile(model_path) or not model_path.lower().endswith(MODEL_EXTENSIONS):
            QMessageBox.warning(self, 'Inference Folder', 'Please select a valid .pt or .onnx model.')
            return None
        if not os.path.isdir(source_folder):
            QMessageBox.warning(self, 'Inference Folder', 'Please select a valid image folder.')
            return None
        if not output_root:
            QMessageBox.warning(self, 'Inference Folder', 'Please select an output root folder.')
            return None
        image_paths = collect_images(source_folder, self.include_subfolders.isChecked())
        if not image_paths:
            QMessageBox.information(self, 'Inference Folder', 'No supported images were found.')
            return None
        return model_path, source_folder, output_root, image_paths

    def _start_inference(self):
        validated = self._validate()
        if validated is None:
            return
        model_path, source_folder, output_root, image_paths = validated
        self._save_settings()

        run_name = '%s_%s' % (
            os.path.splitext(os.path.basename(model_path))[0],
            datetime.now().strftime('%Y%m%d_%H%M%S'))
        output_dir = os.path.join(output_root, run_name)

        self.results_table.setRowCount(0)
        self.results = []
        self.last_output_dir = None
        self.export_button.setEnabled(False)
        self.preview_label.clear()
        self.preview_label.setText('Inference is running...')
        self.progress_bar.setRange(0, len(image_paths))
        self.progress_bar.setValue(0)
        self.status_label.setText('Loading model and processing %d images...' % len(image_paths))
        self._set_running(True)

        self.thread = QThread(self)
        self.worker = InferenceWorker(
            model_path, source_folder, image_paths, output_dir,
            self.confidence_spin.value(), self.iou_spin.value())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.cancel_worker.connect(self.worker.cancel)
        self.worker.progress.connect(self._append_result)
        self.worker.failed.connect(self._inference_failed)
        self.worker.completed.connect(self._inference_completed)
        self.worker.failed.connect(self.thread.quit)
        self.worker.completed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def _append_result(self, result):
        result_index = len(self.results)
        self.results.append(result)
        for detection in result['detection_items']:
            class_name = detection['class_name']
            if self.class_filter.findData(class_name) < 0:
                self.class_filter.addItem(class_name, class_name)
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        values = (
            str(result['index']), result['relative_path'],
            '-' if result['detections'] is None else str(result['detections']),
            '-' if result['max_confidence'] is None
            else '%.3f' % result['max_confidence'],
            '%.0f ms' % result['elapsed_ms'],
            result['status'],
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 0:
                item.setData(Qt.UserRole, result_index)
            if result['error']:
                item.setToolTip(result['error'])
            self.results_table.setItem(row, column, item)
        self.progress_bar.setValue(result['index'])
        self.status_label.setText('Processed %d / %d' % (
            result['index'], self.progress_bar.maximum()))
        self._apply_filter()
        if not self.results_table.isRowHidden(row):
            self.results_table.selectRow(row)

    def _selected_class_name(self):
        value = self.class_filter.itemData(self.class_filter.currentIndex())
        return '' if value is None else str(value)

    def _filtered_detections(self, result):
        return filter_detections(
            result['detection_items'], self._selected_class_name(),
            self.score_filter.value())

    def _apply_filter(self, _value=None):
        visible_images = 0
        visible_detections = 0
        for row, result in enumerate(self.results):
            matches = self._filtered_detections(result)
            show_row = result['status'] == 'Failed' or bool(matches)
            self.results_table.setRowHidden(row, not show_row)
            if result['status'] == 'Completed':
                count_item = self.results_table.item(row, 2)
                score_item = self.results_table.item(row, 3)
                if count_item is not None:
                    count_item.setText(str(len(matches)))
                if score_item is not None:
                    score_item.setText(
                        '-' if not matches
                        else '%.3f' % max(item['score'] for item in matches))
            if matches:
                visible_images += 1
                visible_detections += len(matches)
        self.filter_summary.setText('%d detections in %d images' % (
            visible_detections, visible_images))
        running = self.thread is not None and self.thread.isRunning()
        self.export_button.setEnabled(visible_detections > 0 and not running)
        if self.results_table.currentRow() >= 0:
            self._show_selected_result()

    def _show_selected_result(self):
        row = self.results_table.currentRow()
        if row < 0:
            return
        item = self.results_table.item(row, 0)
        if item is None:
            return
        result_index = item.data(Qt.UserRole)
        if result_index is None:
            return
        result = self.results[int(result_index)]
        if result['status'] != 'Completed':
            self.preview_pixmap = None
            self.preview_label.setText(result['error'])
            return
        try:
            image = _read_image(result['image_path'])
            rendered = render_detections(image, self._filtered_detections(result))
            self.preview_pixmap = cv_image_to_qpixmap(rendered)
            self._refresh_preview()
        except Exception as error:
            self.preview_pixmap = None
            self.preview_label.setText('Could not render preview: %s' % error)

    def _refresh_preview(self):
        if self.preview_pixmap is None:
            return
        self.preview_label.setPixmap(self.preview_pixmap.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _export_filtered_annotations(self):
        if not self.last_output_dir:
            QMessageBox.information(
                self, 'Inference Folder', 'Run inference before exporting annotations.')
            return

        export_dir = os.path.join(
            self.last_output_dir,
            'filtered_annotations_%s' % datetime.now().strftime('%Y%m%d_%H%M%S'))
        exported_files = 0
        exported_objects = 0
        try:
            from libs.pascal_voc_io import PascalVocWriter

            for result in self.results:
                if result['status'] != 'Completed' or not result['image_shape']:
                    continue
                detections = self._filtered_detections(result)
                if not detections:
                    continue

                relative_stem = os.path.splitext(result['relative_path'])[0]
                xml_path = os.path.join(export_dir, relative_stem + '.xml')
                os.makedirs(os.path.dirname(xml_path), exist_ok=True)
                image_path = result['image_path']
                writer = PascalVocWriter(
                    os.path.basename(os.path.dirname(image_path)),
                    os.path.splitext(os.path.basename(image_path))[0],
                    result['image_shape'], localImgPath=image_path)
                for detection in detections:
                    if detection['kind'] == 'obb':
                        writer.addRotatedBndBox(
                            round(detection['cx'], 4), round(detection['cy'], 4),
                            round(detection['width'], 4),
                            round(detection['height'], 4),
                            round(detection['angle'] % math.pi, 6),
                            detection['class_name'], 0)
                    else:
                        writer.addBndBox(
                            round(detection['xmin'], 4),
                            round(detection['ymin'], 4),
                            round(detection['xmax'], 4),
                            round(detection['ymax'], 4),
                            detection['class_name'], 0)
                writer.save(targetFile=xml_path)
                exported_files += 1
                exported_objects += len(detections)
        except Exception as error:
            QMessageBox.critical(
                self, 'Inference Folder',
                'Could not export annotations.\n\n%s: %s' %
                (type(error).__name__, error))
            return

        QMessageBox.information(
            self, 'Inference Folder',
            'Exported %d objects in %d annotation files.\n\n%s' %
            (exported_objects, exported_files, export_dir))

    def resizeEvent(self, event):
        super(InferenceFolderDialog, self).resizeEvent(event)
        self._refresh_preview()

    def _cancel_inference(self):
        self.cancel_button.setEnabled(False)
        self.status_label.setText('Cancelling after the current image...')
        # The worker is busy inside its run slot, so a queued signal alone is
        # not handled until that slot returns. Updating this Python boolean is
        # safe under the GIL and lets the loop stop between images.
        if self.worker is not None:
            self.worker.cancel_requested = True
        self.cancel_worker.emit()

    def _inference_completed(self, summary):
        self.last_output_dir = summary['output_dir']
        self.progress_bar.setValue(summary['processed'])
        if summary['cancelled']:
            message = 'Cancelled: %d / %d images processed.' % (
                summary['processed'], summary['total'])
        else:
            message = 'Completed: %d images, %d failed. Output: %s' % (
                summary['processed'], summary['failed'], summary['output_dir'])
        self.status_label.setText(message)
        self._set_running(False)

    def _inference_failed(self, message):
        self.status_label.setText('Inference failed.')
        self._set_running(False)
        QMessageBox.critical(self, 'Inference Folder', message)

    def _thread_finished(self):
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self._apply_filter()

    def _set_running(self, running):
        for widget in (self.model_combo, self.model_button, self.folder_button,
                       self.output_button, self.include_subfolders,
                       self.confidence_spin, self.iou_spin, self.class_filter,
                       self.score_filter, self.start_button):
            widget.setEnabled(not running)
        if running:
            self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(running)
        self.close_button.setEnabled(not running)

    def closeEvent(self, event):
        self._save_settings()
        if self.thread is not None and self.thread.isRunning():
            self._cancel_inference()
            QMessageBox.information(
                self, 'Inference Folder',
                'Inference is stopping. Close the window after the current image finishes.')
            event.ignore()
            return
        event.accept()

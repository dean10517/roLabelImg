# -*- coding: utf-8 -*-
"""Training configuration, progress, and result viewer for roLabelImg."""

from __future__ import print_function

import json
import os
import re
import sys
from datetime import datetime

try:
    from PyQt5.QtCore import Qt, QProcess, QSettings, QStandardPaths, QUrl
    from PyQt5.QtGui import QDesktopServices, QPixmap
    from PyQt5.QtWidgets import (
        QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog,
        QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
        QPushButton, QSpinBox, QSplitter, QTabWidget, QTableWidget,
        QTableWidgetItem, QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt4.QtCore import Qt, QProcess, QSettings, QUrl
    from PyQt4.QtGui import (
        QAbstractItemView, QCheckBox, QComboBox, QDesktopServices, QDialog,
        QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPixmap,
        QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter,
        QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    )
    QStandardPaths = None

from training_runner import EVENT_PREFIX, scan_annotations


def training_settings_path():
    if QStandardPaths is not None:
        config_dir = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.GenericConfigLocation),
            'roLabelImg')
    else:
        config_dir = os.path.join(os.path.expanduser('~'), '.roLabelImg')
    return os.path.join(config_dir, 'training.ini')


class TrainingDialog(QDialog):
    def __init__(self, project_dir, source_dir=None, parent=None):
        super(TrainingDialog, self).__init__(parent)
        self.project_dir = os.path.abspath(project_dir)
        self.initial_source_dir = os.path.abspath(source_dir) if source_dir else self.project_dir
        self.settings_path = training_settings_path()
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        self.settings = QSettings(self.settings_path, QSettings.IniFormat)
        self.process = None
        self.process_buffer = ''
        self.run_dir = None
        self.cancel_file = None
        self.config_path = None
        self.preflight = None
        self.result_pixmap = None

        self.setWindowTitle('Train Model')
        self.resize(1120, 820)
        self.setModal(True)
        self._build_ui()
        self._load_settings()
        self._model_selection_changed()
        self._onnx_toggled(self.export_onnx.isChecked())
        self._run_preflight(show_dialog=False)

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_settings_tab(), 'Settings')
        self.tabs.addTab(self._build_progress_tab(), 'Progress')
        self.tabs.addTab(self._build_results_tab(), 'Results')

        self.start_button = QPushButton('Start Training')
        self.start_button.clicked.connect(self._start_training)
        self.stop_button = QPushButton('Stop After Current Epoch')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._request_stop)
        self.open_output_button = QPushButton('Open Output Folder')
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch()
        buttons.addWidget(self.open_output_button)
        buttons.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addLayout(buttons)

    def _build_settings_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        dataset_group = QGroupBox('Dataset')
        dataset_grid = QGridLayout(dataset_group)
        self.source_edit = QLineEdit()
        source_button = QPushButton('Browse...')
        source_button.clicked.connect(self._browse_source)
        self.output_edit = QLineEdit()
        output_button = QPushButton('Browse...')
        output_button.clicked.connect(self._browse_output)
        self.include_subfolders = QCheckBox('Include subfolders')
        self.task_combo = QComboBox()
        self.task_combo.addItem('Oriented bounding boxes (OBB)', 'obb')
        self.task_combo.addItem('Axis-aligned detection', 'detect')
        self.task_combo.currentIndexChanged.connect(self._task_changed)
        self.validation_spin = QSpinBox()
        self.validation_spin.setRange(0, 50)
        self.validation_spin.setSuffix('%')
        self.validation_spin.setValue(20)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(42)
        self.preflight_button = QPushButton('Check Annotations')
        self.preflight_button.clicked.connect(self._run_preflight)
        self.preflight_label = QLabel('Not checked')
        self.preflight_label.setWordWrap(True)

        dataset_grid.addWidget(QLabel('Current annotation folder'), 0, 0)
        dataset_grid.addWidget(self.source_edit, 0, 1)
        dataset_grid.addWidget(source_button, 0, 2)
        dataset_grid.addWidget(QLabel('Output root'), 1, 0)
        dataset_grid.addWidget(self.output_edit, 1, 1)
        dataset_grid.addWidget(output_button, 1, 2)
        dataset_grid.addWidget(QLabel('Task'), 2, 0)
        dataset_grid.addWidget(self.task_combo, 2, 1)
        dataset_grid.addWidget(self.include_subfolders, 2, 2)
        dataset_grid.addWidget(QLabel('Validation split'), 3, 0)
        dataset_grid.addWidget(self.validation_spin, 3, 1)
        dataset_grid.addWidget(QLabel('Split seed'), 3, 2)
        dataset_grid.addWidget(self.seed_spin, 3, 3)
        dataset_grid.addWidget(self.preflight_button, 4, 0)
        dataset_grid.addWidget(self.preflight_label, 4, 1, 1, 3)

        self.class_table = QTableWidget(0, 4)
        self.class_table.setHorizontalHeaderLabels(('Use', 'Source class', 'Training class', 'Objects'))
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.class_table.setColumnWidth(0, 55)
        self.class_table.setColumnWidth(1, 230)
        self.class_table.setColumnWidth(2, 230)
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setMinimumHeight(145)

        model_group = QGroupBox('Model and Training')
        model_grid = QGridLayout(model_group)
        self.family_combo = QComboBox()
        self.family_combo.addItems(('YOLO11', 'YOLOv8'))
        self.family_combo.currentIndexChanged.connect(self._model_selection_changed)
        self.size_combo = QComboBox()
        self.size_combo.addItems(('n', 's', 'm', 'l', 'x'))
        self.size_combo.currentIndexChanged.connect(self._model_selection_changed)
        self.model_edit = QLineEdit()
        model_button = QPushButton('Browse .pt...')
        model_button.clicked.connect(self._browse_model)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 10000)
        self.patience_spin.setValue(30)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.batch_combo = QComboBox()
        self.batch_combo.addItem('Auto (60% GPU memory)', -1)
        for value in (1, 2, 4, 8, 16, 32, 64):
            self.batch_combo.addItem(str(value), value)
        self.device_combo = QComboBox()
        self.device_combo.addItem('GPU 0', '0')
        self.device_combo.addItem('CPU', 'cpu')
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(4)
        self.cache_combo = QComboBox()
        self.cache_combo.addItem('Off', False)
        self.cache_combo.addItem('RAM', 'ram')
        self.cache_combo.addItem('Disk', 'disk')
        self.amp_check = QCheckBox('Automatic mixed precision (AMP)')
        self.amp_check.setChecked(True)
        self.rect_check = QCheckBox('Rectangular batches')
        self.deterministic_check = QCheckBox('Deterministic training')
        self.deterministic_check.setChecked(True)

        model_grid.addWidget(QLabel('Family'), 0, 0)
        model_grid.addWidget(self.family_combo, 0, 1)
        model_grid.addWidget(QLabel('Size'), 0, 2)
        model_grid.addWidget(self.size_combo, 0, 3)
        model_grid.addWidget(QLabel('Initial .pt model'), 1, 0)
        model_grid.addWidget(self.model_edit, 1, 1, 1, 2)
        model_grid.addWidget(model_button, 1, 3)
        model_grid.addWidget(QLabel('Epochs'), 2, 0)
        model_grid.addWidget(self.epochs_spin, 2, 1)
        model_grid.addWidget(QLabel('Patience'), 2, 2)
        model_grid.addWidget(self.patience_spin, 2, 3)
        model_grid.addWidget(QLabel('Image size'), 3, 0)
        model_grid.addWidget(self.imgsz_spin, 3, 1)
        model_grid.addWidget(QLabel('Batch'), 3, 2)
        model_grid.addWidget(self.batch_combo, 3, 3)
        model_grid.addWidget(QLabel('Device'), 4, 0)
        model_grid.addWidget(self.device_combo, 4, 1)
        model_grid.addWidget(QLabel('Workers'), 4, 2)
        model_grid.addWidget(self.workers_spin, 4, 3)
        model_grid.addWidget(QLabel('Cache'), 5, 0)
        model_grid.addWidget(self.cache_combo, 5, 1)
        model_grid.addWidget(self.amp_check, 5, 2, 1, 2)
        model_grid.addWidget(self.rect_check, 6, 0, 1, 2)
        model_grid.addWidget(self.deterministic_check, 6, 2, 1, 2)

        export_group = QGroupBox('Output Models')
        export_layout = QHBoxLayout(export_group)
        required = QLabel('best.pt and last.pt are always saved')
        self.export_onnx = QCheckBox('Also export ONNX')
        self.export_onnx.setChecked(True)
        self.export_onnx.toggled.connect(self._onnx_toggled)
        self.onnx_opset = QSpinBox()
        self.onnx_opset.setRange(11, 21)
        self.onnx_opset.setValue(17)
        self.onnx_dynamic = QCheckBox('Dynamic input')
        self.onnx_simplify = QCheckBox('Simplify')
        export_layout.addWidget(required)
        export_layout.addStretch()
        export_layout.addWidget(self.export_onnx)
        export_layout.addWidget(QLabel('Opset'))
        export_layout.addWidget(self.onnx_opset)
        export_layout.addWidget(self.onnx_dynamic)
        export_layout.addWidget(self.onnx_simplify)

        layout.addWidget(dataset_group)
        layout.addWidget(QLabel('Class mapping (clear the training class to exclude it)'))
        layout.addWidget(self.class_table)
        layout.addWidget(model_group)
        layout.addWidget(export_group)
        layout.addStretch()
        return page

    def _build_progress_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.stage_label = QLabel('Ready')
        self.stage_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.metric_label = QLabel('No metrics yet')
        self.metric_label.setWordWrap(True)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.metric_label)
        layout.addWidget(self.log_edit, 1)
        return page

    def _build_results_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.result_summary = QLabel('Training has not been run.')
        self.result_summary.setWordWrap(True)
        splitter = QSplitter(Qt.Horizontal)
        self.artifact_list = QListWidget()
        self.artifact_list.currentItemChanged.connect(self._show_artifact)
        self.result_preview = QLabel('Plots and validation annotations will appear here.')
        self.result_preview.setAlignment(Qt.AlignCenter)
        self.result_preview.setMinimumSize(500, 350)
        splitter.addWidget(self.artifact_list)
        splitter.addWidget(self.result_preview)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(self.result_summary)
        layout.addWidget(splitter, 1)
        return page

    def _load_settings(self):
        self.source_edit.setText(self.initial_source_dir)
        default_output = os.path.join(self.project_dir, 'training_runs')
        self.output_edit.setText(str(self.settings.value('outputRoot', default_output)))
        self.validation_spin.setValue(int(self.settings.value('validationPercent', 20)))
        self.epochs_spin.setValue(int(self.settings.value('epochs', 100)))
        self.patience_spin.setValue(int(self.settings.value('patience', 30)))
        self.imgsz_spin.setValue(int(self.settings.value('imgsz', 640)))
        self.workers_spin.setValue(int(self.settings.value('workers', 4)))

    def _save_settings(self):
        self.settings.setValue('outputRoot', self.output_edit.text().strip())
        self.settings.setValue('validationPercent', self.validation_spin.value())
        self.settings.setValue('epochs', self.epochs_spin.value())
        self.settings.setValue('patience', self.patience_spin.value())
        self.settings.setValue('imgsz', self.imgsz_spin.value())
        self.settings.setValue('workers', self.workers_spin.value())
        self.settings.sync()

    def _browse_source(self):
        path = QFileDialog.getExistingDirectory(
            self, 'Select annotation folder', self.source_edit.text().strip())
        if path:
            self.source_edit.setText(path)
            self._run_preflight(show_dialog=False)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, 'Select output root', self.output_edit.text().strip() or self.project_dir)
        if path:
            self.output_edit.setText(path)

    def _browse_model(self):
        path, _selected = QFileDialog.getOpenFileName(
            self, 'Select initial PyTorch model', self.project_dir,
            'PyTorch model (*.pt)')
        if path:
            self.model_edit.setText(path)

    def _task_changed(self):
        self._model_selection_changed()
        self._run_preflight(show_dialog=False)

    def _model_selection_changed(self):
        family = 'yolo11' if self.family_combo.currentIndex() == 0 else 'yolov8'
        size = self.size_combo.currentText()
        suffix = '-obb' if self.task_combo.currentData() == 'obb' else ''
        current = self.model_edit.text().strip()
        if not current or re.match(r'^yolo(?:11|v8)[nslmx](?:-obb)?\.pt$', current, re.I):
            self.model_edit.setText('%s%s%s.pt' % (family, size, suffix))

    def _onnx_toggled(self, enabled):
        self.onnx_opset.setEnabled(enabled)
        self.onnx_dynamic.setEnabled(enabled)
        self.onnx_simplify.setEnabled(enabled)

    def _existing_mapping(self):
        mapping = {}
        for row in range(self.class_table.rowCount()):
            use = self.class_table.item(row, 0)
            source = self.class_table.item(row, 1)
            target = self.class_table.item(row, 2)
            if source is not None and target is not None:
                mapping[source.text()] = (target.text().strip()
                                          if use.checkState() == Qt.Checked else '')
        return mapping

    def _run_preflight(self, _checked=False, show_dialog=True):
        source_dir = self.source_edit.text().strip()
        if not os.path.isdir(source_dir):
            self.preflight = None
            self.preflight_label.setText('Select a valid annotation folder.')
            if show_dialog:
                QMessageBox.warning(self, 'Train Model', 'Select a valid annotation folder.')
            return False
        previous = self._existing_mapping()
        try:
            self.preflight = scan_annotations(
                source_dir, str(self.task_combo.currentData()),
                self.include_subfolders.isChecked())
        except Exception as error:
            self.preflight = None
            self.preflight_label.setText('%s: %s' % (type(error).__name__, error))
            return False

        counts = self.preflight['label_counts']
        self.class_table.setRowCount(0)
        for label, count in sorted(counts.items()):
            row = self.class_table.rowCount()
            self.class_table.insertRow(row)
            use_item = QTableWidgetItem()
            use_item.setFlags((use_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            use_item.setCheckState(
                Qt.Unchecked if label in previous and not previous[label] else Qt.Checked)
            source_item = QTableWidgetItem(label)
            source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
            target_item = QTableWidgetItem(previous.get(label, label))
            count_item = QTableWidgetItem(str(count))
            count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
            self.class_table.setItem(row, 0, use_item)
            self.class_table.setItem(row, 1, source_item)
            self.class_table.setItem(row, 2, target_item)
            self.class_table.setItem(row, 3, count_item)

        summary = '%d usable images, %d objects, %d classes' % (
            self.preflight['image_count'], self.preflight['object_count'], len(counts))
        if self.preflight['errors']:
            summary += '; %d error(s)' % len(self.preflight['errors'])
            self.preflight_label.setStyleSheet('color: #b00020;')
        elif self.preflight['warnings']:
            summary += '; %d warning(s)' % len(self.preflight['warnings'])
            self.preflight_label.setStyleSheet('color: #9a6700;')
        else:
            self.preflight_label.setStyleSheet('color: #167c2e;')
        self.preflight_label.setText(summary)
        if show_dialog and (self.preflight['errors'] or self.preflight['warnings']):
            details = self.preflight['errors'] + self.preflight['warnings']
            QMessageBox.warning(self, 'Annotation Check', summary + '\n\n' + '\n'.join(details[:20]))
        return not self.preflight['errors'] and bool(self.preflight['records'])

    def _validate_settings(self):
        if not self._run_preflight(show_dialog=True):
            return None
        output_root = self.output_edit.text().strip()
        if not output_root:
            QMessageBox.warning(self, 'Train Model', 'Select an output root folder.')
            return None
        model = self.model_edit.text().strip()
        official = re.match(r'^yolo(?:11|v8)[nslmx](?:-obb)?\.pt$', model, re.I)
        if not (official or (os.path.isfile(model) and model.lower().endswith('.pt'))):
            QMessageBox.warning(
                self, 'Train Model',
                'The initial model must be a local .pt file or a supported official model name.')
            return None
        mapping = self._existing_mapping()
        if not any(mapping.values()):
            QMessageBox.warning(self, 'Train Model', 'At least one training class is required.')
            return None
        image_count = self.preflight['image_count']
        validation_count = int(round(image_count * self.validation_spin.value() / 100.0))
        if self.validation_spin.value() and (image_count < 2 or validation_count >= image_count):
            QMessageBox.warning(self, 'Train Model', 'The split must leave images in both train and validation.')
            return None
        return output_root, model, mapping

    def _start_training(self):
        validated = self._validate_settings()
        if validated is None:
            return
        output_root, model, mapping = validated
        os.makedirs(output_root, exist_ok=True)
        family = self.family_combo.currentText().lower().replace('yolo', 'yolo')
        task = str(self.task_combo.currentData())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = '%s_%s_%s_%s' % (family, self.size_combo.currentText(), task, timestamp)
        run_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', run_name)
        self.run_dir = os.path.abspath(os.path.join(output_root, run_name))
        self.cancel_file = os.path.join(output_root, '.' + run_name + '.cancel')
        self.config_path = os.path.join(output_root, '.' + run_name + '.config.json')
        if os.path.isfile(self.cancel_file):
            os.remove(self.cancel_file)

        config = {
            'source_dir': os.path.abspath(self.source_edit.text().strip()),
            'output_root': os.path.abspath(output_root),
            'run_dir': self.run_dir,
            'cancel_file': self.cancel_file,
            'include_subfolders': self.include_subfolders.isChecked(),
            'task': task,
            'validation_percent': self.validation_spin.value(),
            'seed': self.seed_spin.value(),
            'class_mapping': mapping,
            'model': model,
            'epochs': self.epochs_spin.value(),
            'patience': self.patience_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'batch': self.batch_combo.currentData(),
            'device': str(self.device_combo.currentData()),
            'workers': self.workers_spin.value(),
            'cache': self.cache_combo.currentData(),
            'amp': self.amp_check.isChecked(),
            'rect': self.rect_check.isChecked(),
            'deterministic': self.deterministic_check.isChecked(),
            'export_onnx': self.export_onnx.isChecked(),
            'onnx_opset': self.onnx_opset.value(),
            'onnx_dynamic': self.onnx_dynamic.isChecked(),
            'onnx_simplify': self.onnx_simplify.isChecked(),
        }
        with open(self.config_path, 'w', encoding='utf-8') as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
        self._save_settings()
        self.log_edit.clear()
        self.artifact_list.clear()
        self.result_preview.setText('Training is running...')
        self.result_summary.setText('Training is running. Validation results are not ready yet.')
        self.stage_label.setText('Starting training process...')
        self.metric_label.setText('No metrics yet')
        self.progress_bar.setRange(0, 0)
        self.tabs.setCurrentIndex(1)
        self._set_running(True)

        self.process_buffer = ''
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        runner = os.path.join(self.project_dir, 'training_runner.py')
        self.process.start(sys.executable, [runner, '--config', self.config_path])

    def _set_running(self, running):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.close_button.setEnabled(not running)
        self.open_output_button.setEnabled(bool(self.run_dir) and os.path.isdir(self.run_dir))
        self.tabs.setTabEnabled(0, not running)

    def _read_process_output(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        self.process_buffer += data.replace('\r\n', '\n').replace('\r', '\n')
        lines = self.process_buffer.split('\n')
        self.process_buffer = lines.pop()
        for line in lines:
            if line.startswith(EVENT_PREFIX):
                try:
                    self._handle_event(json.loads(line[len(EVENT_PREFIX):]))
                except Exception:
                    self.log_edit.appendPlainText(line)
            elif line.strip():
                self.log_edit.appendPlainText(line)

    def _handle_event(self, event):
        event_type = event.get('type')
        if event_type == 'stage':
            self.stage_label.setText(event.get('message', event.get('name', 'Working...')))
            self.progress_bar.setRange(0, 0)
        elif event_type == 'prepare_progress':
            self.stage_label.setText('Preparing dataset: %d / %d' %
                                     (event['current'], event['total']))
            self.progress_bar.setRange(0, event['total'])
            self.progress_bar.setValue(event['current'])
        elif event_type == 'dataset_ready':
            self.log_edit.appendPlainText(
                'Dataset ready: %d train, %d validation; classes: %s' %
                (event['train'], event['val'], ', '.join(event['classes'])))
        elif event_type == 'epoch':
            current, total = event['current'], event['total']
            self.stage_label.setText('Training epoch %d / %d' % (current, total))
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            metrics = event.get('metrics') or {}
            preferred = []
            for key, value in metrics.items():
                if any(token in key.lower() for token in ('map', 'precision', 'recall', 'fitness')):
                    try:
                        preferred.append('%s=%.4f' % (key, float(value)))
                    except Exception:
                        preferred.append('%s=%s' % (key, value))
            self.metric_label.setText(' | '.join(preferred) if preferred else 'Epoch %d completed' % current)
        elif event_type == 'warning':
            self.log_edit.appendPlainText('WARNING: ' + event.get('message', ''))
        elif event_type == 'completed':
            message = 'Completed. best.pt: %s' % (event.get('best_pt') or 'not produced')
            if self.validation_spin.value() == 0:
                message += ('\nNo independent validation set was configured; '
                            'mAP and confusion matrix are not reported as validation results.')
            if event.get('export_error'):
                message += '\nONNX export warning: ' + event['export_error']
            self.stage_label.setText('Training completed')
            self.result_summary.setText(message)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self._load_artifacts()
            self.tabs.setCurrentIndex(2)
        elif event_type == 'cancelled':
            self.stage_label.setText(event.get('message', 'Training cancelled.'))
            self.result_summary.setText('Training was cancelled. A checkpoint from the last completed epoch may be available.')
            self._load_artifacts()
        elif event_type == 'failed':
            self.stage_label.setText('Training failed')
            self.log_edit.appendPlainText(event.get('traceback', event.get('message', 'Unknown error')))
            QMessageBox.critical(self, 'Train Model', event.get('message', 'Training failed.'))

    def _request_stop(self):
        if not self.cancel_file:
            return
        try:
            with open(self.cancel_file, 'w', encoding='utf-8') as stream:
                stream.write('cancel requested\n')
            self.stop_button.setEnabled(False)
            self.stage_label.setText('Stop requested; waiting for a safe checkpoint...')
        except Exception as error:
            QMessageBox.critical(self, 'Train Model', 'Could not request stop: %s' % error)

    def _process_finished(self, exit_code, _exit_status):
        if self.process is None:
            return
        self._read_process_output()
        if self.process_buffer.strip():
            self.log_edit.appendPlainText(self.process_buffer.strip())
        self.process_buffer = ''
        self._set_running(False)
        self.open_output_button.setEnabled(bool(self.run_dir) and os.path.isdir(self.run_dir))
        if exit_code not in (0, 2) and self.stage_label.text() != 'Training failed':
            self.stage_label.setText('Training process exited with code %d' % exit_code)
        self.process.deleteLater()
        self.process = None
        self._cleanup_control_files()

    def _process_error(self, error):
        if self.process is not None and self.process.state() == QProcess.NotRunning:
            self.stage_label.setText('Could not start training process: %s' % error)
            self._set_running(False)
            self._cleanup_control_files()
            self.process.deleteLater()
            self.process = None

    def _cleanup_control_files(self):
        for path in (self.config_path, self.cancel_file):
            try:
                if path and os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass

    def _load_artifacts(self):
        self.artifact_list.clear()
        if not self.run_dir or not os.path.isdir(self.run_dir):
            return
        preferred = ('confusion_matrix', 'results.png', 'curve',
                     'val_batch', 'validation')
        artifacts = []
        for root, _dirs, files in os.walk(self.run_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, filename)
                    name = os.path.relpath(path, self.run_dir)
                    if self.validation_spin.value() == 0 and any(
                            token in name.lower()
                            for token in ('confusion', 'curve', 'val_batch', 'validation')):
                        continue
                    if any(token in name.lower() for token in preferred):
                        artifacts.append((name, path))
        for name, path in sorted(artifacts, key=lambda item: item[0].lower()):
            widget_item = QListWidgetItem(name)
            widget_item.setData(Qt.UserRole, path)
            self.artifact_list.addItem(widget_item)
        if self.artifact_list.count():
            self.artifact_list.setCurrentRow(0)
        elif self.validation_spin.value() == 0:
            self.result_preview.setText(
                'No independent validation set was configured.\nConfusion matrix and validation annotations are unavailable.')
        else:
            self.result_preview.setText('No result images were found.')

    def _show_artifact(self, current, _previous=None):
        if current is None:
            return
        path = current.data(Qt.UserRole)
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.result_pixmap = None
            self.result_preview.setText('Could not load: ' + str(path))
            return
        self.result_pixmap = pixmap
        self._refresh_preview()

    def _refresh_preview(self):
        if self.result_pixmap is not None:
            self.result_preview.setPixmap(self.result_pixmap.scaled(
                self.result_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _open_output_folder(self):
        path = self.run_dir if self.run_dir and os.path.isdir(self.run_dir) else self.output_edit.text().strip()
        if path and os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def resizeEvent(self, event):
        super(TrainingDialog, self).resizeEvent(event)
        self._refresh_preview()

    def closeEvent(self, event):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(
                self, 'Train Model',
                'Training is still running. Request a safe stop before closing this window.')
            event.ignore()
            return
        self._save_settings()
        self._cleanup_control_files()
        event.accept()

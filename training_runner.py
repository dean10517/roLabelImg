# -*- coding: utf-8 -*-
"""Standalone dataset preparation and Ultralytics training process."""

from __future__ import print_function

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime


IMAGE_EXTENSIONS = ('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp')
EVENT_PREFIX = 'ROTRAIN_EVENT '


class TrainingCancelled(Exception):
    pass


def emit(event_type, **values):
    payload = dict(values)
    payload['type'] = event_type
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _find_image(xml_path):
    stem = os.path.splitext(xml_path)[0]
    for extension in IMAGE_EXTENSIONS:
        for candidate in (stem + extension, stem + extension.upper()):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return None


def _rotated_points(cx, cy, width, height, angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    points = []
    for x, y in ((-width / 2.0, -height / 2.0),
                 (width / 2.0, -height / 2.0),
                 (width / 2.0, height / 2.0),
                 (-width / 2.0, height / 2.0)):
        points.append((cx + cosine * x - sine * y,
                       cy + sine * x + cosine * y))
    return points


def _annotation_paths(source_dir, include_subfolders):
    if include_subfolders:
        paths = []
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [item for item in dirs if item not in ('training_runs', '.git')]
            paths.extend(os.path.join(root, item) for item in files
                         if item.lower().endswith('.xml'))
        return sorted(paths, key=lambda value: value.lower())
    return sorted(
        (os.path.join(source_dir, item) for item in os.listdir(source_dir)
         if item.lower().endswith('.xml')),
        key=lambda value: value.lower())


def scan_annotations(source_dir, task='obb', include_subfolders=False):
    """Return parsed records and a user-facing preflight summary."""
    source_dir = os.path.abspath(source_dir)
    records = []
    errors = []
    warnings = []
    label_counts = Counter()
    annotation_paths = _annotation_paths(source_dir, include_subfolders)
    expected_tag = 'robndbox' if task == 'obb' else 'bndbox'
    other_tag = 'bndbox' if task == 'obb' else 'robndbox'

    for xml_path in annotation_paths:
        try:
            root = ET.parse(xml_path).getroot()
            width = int(float(root.findtext('size/width', '0')))
            height = int(float(root.findtext('size/height', '0')))
            if width <= 0 or height <= 0:
                raise ValueError('invalid image width or height')
            image_path = _find_image(xml_path)
            if image_path is None:
                raise ValueError('matching image file was not found')
            objects = []
            ignored_other = 0
            for item in root.findall('object'):
                label = (item.findtext('name') or '').strip()
                box = item.find(expected_tag)
                if box is None:
                    if item.find(other_tag) is not None:
                        ignored_other += 1
                    continue
                if not label:
                    raise ValueError('object has an empty class name')
                if task == 'obb':
                    values = [float(box.findtext(name, 'nan'))
                              for name in ('cx', 'cy', 'w', 'h', 'angle')]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError('rotated box contains a non-finite value')
                    cx, cy, box_width, box_height, angle = values
                    if box_width <= 0 or box_height <= 0:
                        raise ValueError('rotated box has a non-positive size')
                    points = _rotated_points(cx, cy, box_width, box_height, angle)
                    normalized = [(x / width, y / height) for x, y in points]
                    if any(x < 0 or x > 1 or y < 0 or y > 1
                           for x, y in normalized):
                        raise ValueError('rotated box extends outside the image')
                    coordinates = [value for point in normalized for value in point]
                else:
                    values = [float(box.findtext(name, 'nan'))
                              for name in ('xmin', 'ymin', 'xmax', 'ymax')]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError('box contains a non-finite value')
                    xmin, ymin, xmax, ymax = values
                    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                        raise ValueError('box extends outside the image')
                    if xmax <= xmin or ymax <= ymin:
                        raise ValueError('box has a non-positive size')
                    coordinates = [
                        ((xmin + xmax) / 2.0) / width,
                        ((ymin + ymax) / 2.0) / height,
                        (xmax - xmin) / width,
                        (ymax - ymin) / height,
                    ]
                objects.append({'label': label, 'coordinates': coordinates})
                label_counts[label] += 1
            if ignored_other:
                warnings.append('%s: ignored %d %s object(s)' %
                                (os.path.basename(xml_path), ignored_other, other_tag))
            if not objects:
                if ignored_other:
                    warnings.append('%s: skipped because it has no %s objects' %
                                    (os.path.basename(xml_path), expected_tag))
                    continue
                raise ValueError('no usable %s objects' % expected_tag)
            records.append({
                'xml_path': os.path.abspath(xml_path),
                'image_path': image_path,
                'relative_path': os.path.relpath(image_path, source_dir),
                'width': width,
                'height': height,
                'objects': objects,
            })
        except Exception as error:
            errors.append('%s: %s' % (os.path.basename(xml_path), error))

    if not annotation_paths:
        errors.append('No XML annotation files were found.')
    return {
        'source_dir': source_dir,
        'task': task,
        'records': records,
        'annotation_count': len(annotation_paths),
        'image_count': len(records),
        'object_count': sum(label_counts.values()),
        'label_counts': dict(sorted(label_counts.items())),
        'errors': errors,
        'warnings': warnings,
    }


def _copy_file(source, target):
    parent = os.path.dirname(target)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    shutil.copy2(source, target)


def prepare_dataset(config, preflight=None):
    if preflight is None:
        preflight = scan_annotations(
            config['source_dir'], config['task'], config.get('include_subfolders', False))
    if preflight['errors']:
        raise ValueError('\n'.join(preflight['errors']))

    mapping = config.get('class_mapping', {})
    records = []
    for record in preflight['records']:
        objects = []
        for item in record['objects']:
            target = mapping.get(item['label'], item['label']).strip()
            if target:
                copied = dict(item)
                copied['target_label'] = target
                objects.append(copied)
        if objects:
            copied_record = dict(record)
            copied_record['objects'] = objects
            records.append(copied_record)
    if not records:
        raise ValueError('Every annotated object was excluded by the class mapping.')

    class_names = sorted(set(item['target_label'] for record in records
                             for item in record['objects']))
    class_ids = dict((name, index) for index, name in enumerate(class_names))
    seed = int(config.get('seed', 42))
    random.Random(seed).shuffle(records)
    val_ratio = float(config.get('validation_percent', 20)) / 100.0
    val_count = 0 if val_ratio <= 0 else max(1, int(round(len(records) * val_ratio)))
    if val_count >= len(records):
        raise ValueError('The validation split leaves no training images.')
    validation_keys = set(item['image_path'] for item in records[:val_count])
    dataset_dir = os.path.join(config['run_dir'], 'dataset')
    manifest_rows = ['split,image,annotation,objects']
    split_counts = Counter()

    for index, record in enumerate(records, 1):
        if os.path.isfile(config.get('cancel_file', '')):
            raise TrainingCancelled('Cancelled while preparing the dataset.')
        split = 'val' if record['image_path'] in validation_keys else 'train'
        image_name = os.path.basename(record['image_path'])
        stem = os.path.splitext(image_name)[0]
        if config.get('include_subfolders'):
            prefix = re.sub(r'[^A-Za-z0-9_.-]+', '_',
                            os.path.dirname(record['relative_path'])).strip('_')
            if prefix:
                image_name = prefix + '__' + image_name
                stem = os.path.splitext(image_name)[0]
        image_target = os.path.join(dataset_dir, 'images', split, image_name)
        label_target = os.path.join(dataset_dir, 'labels', split, stem + '.txt')
        _copy_file(record['image_path'], image_target)
        lines = []
        for item in record['objects']:
            values = ' '.join('%.8f' % value for value in item['coordinates'])
            lines.append('%d %s' % (class_ids[item['target_label']], values))
        parent = os.path.dirname(label_target)
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with open(label_target, 'w', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')
        manifest_rows.append('%s,%s,%s,%d' %
                             (split, record['image_path'], record['xml_path'], len(lines)))
        split_counts[split] += 1
        emit('prepare_progress', current=index, total=len(records))

    # Ultralytics expects a val path even when per-epoch validation is disabled.
    val_path = 'images/val' if val_count else 'images/train'
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as stream:
        stream.write('path: %s\n' % json.dumps(dataset_dir.replace('\\', '/')))
        stream.write('train: images/train\n')
        stream.write('val: %s\n' % val_path)
        stream.write('names:\n')
        for index, name in enumerate(class_names):
            stream.write('  %d: %s\n' % (index, json.dumps(name, ensure_ascii=False)))
    with open(os.path.join(config['run_dir'], 'dataset_manifest.csv'),
              'w', encoding='utf-8-sig') as stream:
        stream.write('\n'.join(manifest_rows) + '\n')
    return yaml_path, dict(split_counts), class_names


def _json_scalar(value):
    try:
        if hasattr(value, 'item'):
            value = value.item()
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return float(value)
    except Exception:
        return str(value)


def save_confusion_matrix_artifacts(metrics, output_dir, class_names):
    """Save CSV plus OpenCV heatmaps when Ultralytics plotting is unavailable."""
    import csv
    import cv2
    import numpy as np

    matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=np.float64)
    labels = list(class_names)
    if matrix.shape[0] == len(labels) + 1:
        labels.append('background')
    while len(labels) < matrix.shape[0]:
        labels.append('class_%d' % len(labels))
    labels = labels[:matrix.shape[0]]
    csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['predicted / true'] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + ['%.8g' % value for value in row])

    column_totals = matrix.sum(axis=0, keepdims=True)
    normalized = np.divide(matrix, column_totals, out=np.zeros_like(matrix),
                           where=column_totals != 0)

    def render(values, path, normalized_values):
        count = len(labels)
        cell = max(58, min(105, 700 // max(1, count)))
        left, top, right, bottom = 190, 75, 30, 155
        grid_width, grid_height = count * cell, count * cell
        canvas = np.full((top + grid_height + bottom,
                          left + grid_width + right, 3), 255, dtype=np.uint8)
        maximum = float(values.max()) if values.size else 0.0
        scaled = np.zeros_like(values, dtype=np.uint8)
        if maximum > 0:
            scaled = np.clip(values / maximum * 255.0, 0, 255).astype(np.uint8)
        heatmap = np.stack((
            np.full_like(scaled, 255),
            255 - (scaled.astype(np.float32) * 0.62).astype(np.uint8),
            255 - scaled,
        ), axis=2)
        heatmap = cv2.resize(heatmap, (grid_width, grid_height),
                             interpolation=cv2.INTER_NEAREST)
        canvas[top:top + grid_height, left:left + grid_width] = heatmap
        for index in range(count + 1):
            x = left + index * cell
            y = top + index * cell
            cv2.line(canvas, (x, top), (x, top + grid_height), (220, 220, 220), 1)
            cv2.line(canvas, (left, y), (left + grid_width, y), (220, 220, 220), 1)
        for row in range(count):
            for column in range(count):
                value = values[row, column]
                text_value = ('%.2f' % value) if normalized_values else ('%.0f' % value)
                text_size = cv2.getTextSize(text_value, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                center_x = left + column * cell + cell // 2
                center_y = top + row * cell + cell // 2
                color = (255, 255, 255) if scaled[row, column] > 145 else (25, 25, 25)
                cv2.putText(canvas, text_value,
                            (center_x - text_size[0] // 2, center_y + text_size[1] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        for index, label in enumerate(labels):
            short_label = label if len(label) <= 22 else label[:19] + '...'
            cv2.putText(canvas, short_label, (8, top + index * cell + cell // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas, short_label,
                        (left + index * cell + 4, top + grid_height + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
        title = 'Normalized Confusion Matrix' if normalized_values else 'Confusion Matrix'
        cv2.putText(canvas, title, (left, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Rows: predicted   Columns: true',
                    (left, top + grid_height + 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (60, 60, 60), 1, cv2.LINE_AA)
        success, encoded = cv2.imencode('.png', canvas)
        if not success:
            raise IOError('Could not encode confusion matrix image.')
        encoded.tofile(path)

    normal_path = os.path.join(output_dir, 'confusion_matrix.png')
    normalized_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    if not os.path.isfile(normal_path):
        render(matrix, normal_path, False)
    if not os.path.isfile(normalized_path):
        render(normalized, normalized_path, True)


def run_training(config):
    run_dir = os.path.abspath(config['run_dir'])
    os.makedirs(run_dir, exist_ok=False)
    config['run_dir'] = run_dir
    with open(os.path.join(run_dir, 'run_config.json'), 'w', encoding='utf-8') as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)

    emit('stage', name='preflight', message='Checking annotations...')
    preflight = scan_annotations(
        config['source_dir'], config['task'], config.get('include_subfolders', False))
    if preflight['errors']:
        raise ValueError('\n'.join(preflight['errors']))
    emit('preflight', images=preflight['image_count'], objects=preflight['object_count'],
         labels=preflight['label_counts'], warnings=preflight['warnings'])
    emit('stage', name='prepare', message='Creating a dataset snapshot...')
    yaml_path, split_counts, class_names = prepare_dataset(config, preflight)
    emit('dataset_ready', train=split_counts.get('train', 0),
         val=split_counts.get('val', 0), classes=class_names, yaml=yaml_path)
    if os.path.isfile(config.get('cancel_file', '')):
        raise TrainingCancelled('Cancelled before training started.')

    emit('stage', name='training', message='Loading model and starting training...')
    from ultralytics import YOLO

    model = YOLO(config['model'])
    if str(getattr(model, 'task', '')) != config['task']:
        raise ValueError(
            'The selected model task is %s, but the dataset task is %s.' %
            (getattr(model, 'task', 'unknown'), config['task']))

    callback_state = {'last_epoch': 0}

    def on_epoch_end(trainer):
        current_epoch = int(trainer.epoch) + 1
        metrics = dict((str(key), _json_scalar(value))
                       for key, value in getattr(trainer, 'metrics', {}).items())
        if current_epoch != callback_state['last_epoch']:
            callback_state['last_epoch'] = current_epoch
            emit('epoch', current=current_epoch,
                 total=int(getattr(trainer, 'epochs', config['epochs'])), metrics=metrics)
        if os.path.isfile(config.get('cancel_file', '')):
            raise TrainingCancelled('Training stopped after epoch %d.' % current_epoch)

    model.add_callback('on_fit_epoch_end', on_epoch_end)
    train_args = {
        'data': yaml_path,
        'epochs': int(config['epochs']),
        'patience': int(config['patience']),
        'imgsz': int(config['imgsz']),
        'batch': config['batch'],
        'device': config['device'],
        'workers': int(config['workers']),
        'cache': config['cache'],
        'seed': int(config['seed']),
        'deterministic': bool(config['deterministic']),
        'amp': bool(config['amp']),
        'rect': bool(config['rect']),
        'val': int(config['validation_percent']) > 0,
        'plots': True,
        'project': os.path.dirname(run_dir),
        'name': os.path.basename(run_dir),
        'exist_ok': True,
        'verbose': True,
    }
    model.train(**train_args)

    weights_dir = os.path.join(run_dir, 'weights')
    best_pt = os.path.join(weights_dir, 'best.pt')
    last_pt = os.path.join(weights_dir, 'last.pt')
    selected_pt = best_pt if os.path.isfile(best_pt) else last_pt
    if not os.path.isfile(selected_pt):
        raise RuntimeError('Training finished without producing best.pt or last.pt.')

    validation_dir = None
    if int(config['validation_percent']) > 0:
        emit('stage', name='validation', message='Generating validation metrics and plots...')
        validation_model = YOLO(selected_pt)
        validation_batch = 1 if config['batch'] == -1 else config['batch']
        validation_metrics = validation_model.val(
            data=yaml_path, imgsz=int(config['imgsz']), batch=validation_batch,
            device=config['device'], workers=int(config['workers']), plots=True,
            project=run_dir, name='validation', exist_ok=True, verbose=True)
        validation_dir = os.path.join(run_dir, 'validation')
        try:
            save_confusion_matrix_artifacts(
                validation_metrics, validation_dir, class_names)
        except Exception as error:
            emit('warning', message='Could not save confusion matrix fallback: %s' % error)

    onnx_path = None
    export_error = None
    if config.get('export_onnx'):
        emit('stage', name='export', message='Exporting ONNX...')
        try:
            export_model = YOLO(selected_pt)
            generated = export_model.export(
                format='onnx', imgsz=int(config['imgsz']),
                opset=int(config.get('onnx_opset', 17)),
                dynamic=bool(config.get('onnx_dynamic', False)),
                simplify=bool(config.get('onnx_simplify', False)),
                device=config['device'])
            export_dir = os.path.join(run_dir, 'export')
            os.makedirs(export_dir, exist_ok=True)
            onnx_path = os.path.join(export_dir, 'best.onnx')
            shutil.copy2(os.path.abspath(str(generated)), onnx_path)
            try:
                import onnx
                exported_model = onnx.load(onnx_path)
                onnx.checker.check_model(exported_model)
            except ImportError:
                emit('warning', message='ONNX was exported, but the onnx checker is not installed.')
        except Exception as error:
            export_error = '%s: %s' % (type(error).__name__, error)
            emit('warning', message='ONNX export failed: ' + export_error)

    metadata = {
        'completed_at': datetime.now().isoformat(),
        'source_dir': config['source_dir'],
        'task': config['task'],
        'model': config['model'],
        'classes': class_names,
        'train_images': split_counts.get('train', 0),
        'validation_images': split_counts.get('val', 0),
        'best_pt': best_pt if os.path.isfile(best_pt) else None,
        'last_pt': last_pt if os.path.isfile(last_pt) else None,
        'onnx': onnx_path,
        'onnx_export_error': export_error,
    }
    with open(os.path.join(run_dir, 'training_metadata.json'),
              'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    emit('completed', run_dir=run_dir, best_pt=metadata['best_pt'],
         last_pt=metadata['last_pt'], onnx=onnx_path,
         validation_dir=validation_dir, export_error=export_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as stream:
            config = json.load(stream)
        run_training(config)
        return 0
    except TrainingCancelled as error:
        emit('cancelled', message=str(error))
        return 2
    except Exception as error:
        emit('failed', message='%s: %s' % (type(error).__name__, error),
             traceback=traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())

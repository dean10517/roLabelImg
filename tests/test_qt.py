
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from roLabelImg import MainWindow, Shape
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QImage, QWheelEvent
from PyQt5.QtWidgets import QApplication


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # Never read or overwrite the user's application preferences.
        with patch('roLabelImg.Settings', return_value={}):
            self.win = MainWindow(defaultPrefdefClassFile='')
        self.win.resize(1100, 700)
        self.win.show()
        self.app.processEvents()
        self.paths = []
        for name in ('first', 'second', 'third'):
            path = os.path.join(self.temp.name, name + '.png')
            image = QImage(1600, 1200, QImage.Format_RGB32)
            image.fill(Qt.white)
            self.assertTrue(image.save(path))
            self.paths.append(path)
        self.assertTrue(self.win.loadFile(self.paths[0]))
        self.app.processEvents()

    def tearDown(self):
        self.win.dirty = False
        self.win.close()
        self.win.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def add_box(self, label, rotated=False):
        shape = Shape(label)
        shape.points = [QPointF(100, 100), QPointF(220, 100),
                        QPointF(220, 180), QPointF(100, 180)]
        shape.close()
        shape.isRotated = rotated
        shape.difficult = True
        if rotated:
            shape.rotate(0.2)
        self.win.canvas.shapes.append(shape)
        self.win.addLabel(shape)
        return shape

    def wheel_zoom(self, delta, viewport_point):
        canvas = self.win.canvas
        position = QPointF(viewport_point) - QPointF(canvas.pos())
        before = canvas.transformPos(position)
        event = QWheelEvent(position, QPointF(canvas.mapToGlobal(position.toPoint())),
                            QPoint(), QPoint(0, delta), delta, Qt.Vertical,
                            Qt.NoButton, Qt.ControlModifier)
        canvas.wheelEvent(event)
        self.app.processEvents()
        after = canvas.transformPos(QPointF(viewport_point) - QPointF(canvas.pos()))
        self.assertLess((after - before).manhattanLength() * canvas.scale, 2.1,
                        (canvas.scale, before, after, canvas.size(), canvas.pos()))

    def test_zoom_preserves_cursor_image_point(self):
        self.win.setZoom(100)
        self.app.processEvents()
        for bar in self.win.scrollBars.values():
            bar.setValue(200)
        for delta in (120, 120, -120, -120):
            self.wheel_zoom(delta, QPoint(180, 160))

    def test_zoom_centering_transition_and_limits(self):
        # Keep the viewport size constant as the image crosses its edges.
        scroll = self.win.centralWidget()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.app.processEvents()
        self.win.setZoom(10)
        self.app.processEvents()
        center = self.win.centralWidget().viewport().rect().center()
        for delta in (120, 120, 120, 120, -120):
            self.wheel_zoom(delta, center)
        self.win.setZoom(500)
        self.app.processEvents()
        self.wheel_zoom(120, center)
        self.assertEqual(self.win.zoomWidget.value(), 500)
        self.win.setZoom(1)
        self.app.processEvents()
        self.wheel_zoom(-120, center)
        self.assertEqual(self.win.zoomWidget.value(), 1)

    def test_paste_preserves_all_boxes_and_independent_geometry(self):
        originals = [self.add_box('normal'), self.add_box('rotated', True)]
        self.assertTrue(self.win.loadFile(self.paths[1]))
        existing = self.add_box('existing')
        self.assertTrue(self.win.actions.pastePrevious.isEnabled())
        self.assertEqual(self.win.actions.pastePrevious.shortcut().toString(), 'Ctrl+Q')
        self.assertEqual(self.win.actions.fileMenuActions[-1].shortcut().toString(),
                         'Ctrl+Shift+Q')
        self.win.actions.pastePrevious.trigger()
        self.assertEqual(self.win.canvas.shapes[0], existing)
        self.assertEqual(self.win.labelList.count(), 3)
        self.assertTrue(self.win.dirty)
        self.assertTrue(self.win.actions.save.isEnabled())
        for original, pasted in zip(originals, self.win.canvas.shapes[1:]):
            self.assertEqual(original.points, pasted.points)
            self.assertEqual(original.center, pasted.center)
            self.assertEqual(original.direction, pasted.direction)
            self.assertEqual(original.isRotated, pasted.isRotated)
            self.assertEqual(original.label, pasted.label)
            self.assertEqual(original.difficult, pasted.difficult)
            pasted.points[0].setX(999)
            pasted.center.setX(888)
            self.assertNotEqual(original.points, pasted.points)
            self.assertNotEqual(original.center, pasted.center)
        self.win.pastePreviousShapes()
        self.assertEqual(len(self.win.canvas.shapes), 5)
        self.assertEqual(self.win.canvas.shapes[3].points, originals[0].points)

    def test_empty_previous_image_and_no_image(self):
        self.assertFalse(self.win.actions.pastePrevious.isEnabled())
        self.add_box('source')
        self.win.loadFile(self.paths[1])
        self.assertTrue(self.win.actions.pastePrevious.isEnabled())
        self.win.loadFile(self.paths[2])
        self.assertFalse(self.win.actions.pastePrevious.isEnabled())
        self.win.pastePreviousShapes()
        self.assertFalse(self.win.dirty)
        self.win.closeFile()
        self.assertFalse(self.win.actions.pastePrevious.isEnabled())

    def test_paste_during_drawing_does_not_modify_annotations(self):
        self.add_box('source')
        self.win.loadFile(self.paths[1])
        self.win.createShape()
        self.win.pastePreviousShapes()
        self.assertEqual(self.win.canvas.shapes, [])
        self.assertFalse(self.win.dirty)

    def test_navigation_and_paste_can_be_saved(self):
        self.win.mImgList = self.paths
        self.add_box('source')
        self.win.openNextImg()
        self.win.pastePreviousShapes()
        xml_path = os.path.splitext(self.paths[1])[0] + '.xml'
        self.assertTrue(self.win._saveFile(xml_path))
        self.assertFalse(self.win.dirty)
        self.win.openPrevImg()
        self.win.pastePreviousShapes()
        self.assertEqual(self.win.canvas.shapes[0].label, 'source')
        self.win.dirty = False
        self.win.openNextImg()
        self.assertEqual(len(self.win.canvas.shapes), 1)
        self.assertEqual(self.win.canvas.shapes[0].points,
                         [QPointF(100, 100), QPointF(220, 100),
                          QPointF(220, 180), QPointF(100, 180)])


if __name__ == '__main__':
    unittest.main()

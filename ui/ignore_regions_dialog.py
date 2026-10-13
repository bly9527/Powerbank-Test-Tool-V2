"""Edit non-destructive, crop-relative template exclusion rectangles."""
import cv2
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
                             QPushButton, QDialogButtonBox)
from core.segment_dataset import feature, validate_ignore_regions, mask_ocr_regions


def pixmap(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                   rgb.strides[0], QImage.Format_RGB888).copy())


class IgnoreRegionsCanvas(QWidget):
    changed = pyqtSignal()

    def __init__(self, image, regions=(), parent=None):
        super().__init__(parent)
        self.image = image.copy()
        self.picture = pixmap(image)
        self.regions = list(validate_ignore_regions(regions))
        self.start = self.end = None
        self.setMinimumSize(240, 160)

    def image_rect(self):
        h, w = self.image.shape[:2]
        scale = min(self.width()/w, self.height()/h)
        return QRectF((self.width()-w*scale)/2, (self.height()-h*scale)/2, w*scale, h*scale)

    def normalized_point(self, pos):
        rect = self.image_rect()
        return QPointF(max(0., min(1., (pos.x()-rect.x())/rect.width())),
                       max(0., min(1., (pos.y()-rect.y())/rect.height())))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image_rect().contains(QPointF(event.pos())):
            self.start = self.end = self.normalized_point(event.pos())
            self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = self.normalized_point(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self.start is None:
            return
        end = self.normalized_point(event.pos())
        x1, x2 = sorted((self.start.x(), end.x()))
        y1, y2 = sorted((self.start.y(), end.y()))
        h, w = self.image.shape[:2]
        if (x2-x1)*w >= 1 and (y2-y1)*h >= 1 and len(self.regions) < 32:
            self.regions.append((x1, y1, x2, y2))
            self.changed.emit()
        self.start = self.end = None
        self.update()

    def undo(self):
        if self.regions:
            self.regions.pop()
            self.changed.emit()
            self.update()

    def clear_regions(self):
        self.regions.clear()
        self.changed.emit()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(25, 25, 30))
        rect = self.image_rect()
        painter.drawPixmap(rect, self.picture, QRectF(self.picture.rect()))
        pen = QPen(QColor(255, 90, 60), 1, Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 70, 30, 65))
        regions = list(self.regions)
        if self.start is not None:
            regions.append((min(self.start.x(), self.end.x()), min(self.start.y(), self.end.y()),
                            max(self.start.x(), self.end.x()), max(self.start.y(), self.end.y())))
        for x1, y1, x2, y2 in regions:
            painter.drawRect(QRectF(rect.x()+x1*rect.width(), rect.y()+y1*rect.height(),
                                   (x2-x1)*rect.width(), (y2-y1)*rect.height()))


class IgnoreRegionsDialog(QDialog):
    def __init__(self, image, regions=(), parent=None, lcd_unit=None):
        super().__init__(parent)
        self.lcd_unit = lcd_unit
        self.setWindowTitle(f'{lcd_unit}# LCD 电量图标忽略区域' if lcd_unit else '设置当前模板的图标忽略区域')
        layout = QVBoxLayout(self)
        hint = QLabel('在左侧拖框圈住闪电、百分号等图标（最多 32 个），不要遮住数字。\n'
                      + (f'只对 {lcd_unit}# 的 LCD 电量识别生效，测试当前帧与正式分析都会使用。换屏幕或重新框选后请检查设置。'
                         if lcd_unit else '对当前模板的所有样本生效；采集和识别时请保持数字框的位置与范围一致。原始图片不会被修改。'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        body = QHBoxLayout()
        self.canvas = IgnoreRegionsCanvas(image, regions)
        body.addWidget(self.canvas, 1)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(156, 108)
        body.addWidget(self.preview)
        layout.addLayout(body, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        actions = QHBoxLayout()
        undo = QPushButton('撤销上一个框')
        undo.clicked.connect(self.canvas.undo)
        clear = QPushButton('清除全部忽略区域')
        clear.clicked.connect(self.canvas.clear_regions)
        actions.addWidget(undo)
        actions.addWidget(clear)
        layout.addLayout(actions)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Save).setText(f'保存到 {lcd_unit}# LCD' if lcd_unit else '保存到当前模板')
        self.buttons.button(QDialogButtonBox.Cancel).setText('取消')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.canvas.changed.connect(self.update_preview)
        self.update_preview()
        self.resize(720, 440)

    def update_preview(self):
        result = (mask_ocr_regions(self.canvas.image, self.canvas.regions) if self.lcd_unit
                  else feature(self.canvas.image, self.canvas.regions))
        self.preview.clear()
        if result is None:
            self.preview.setText('当前区域没有\n可匹配的数字')
        else:
            self.preview.setPixmap(pixmap(result).scaled(208, 144, Qt.KeepAspectRatio, Qt.FastTransformation))
        self.info.setText(f'已设置 {len(self.canvas.regions)} 个忽略区域。右侧是处理后的数字图像，请确认数字完整。')
        self.buttons.button(QDialogButtonBox.Save).setEnabled(result is not None or not self.canvas.regions)

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QImage, QPixmap
import numpy as np


class VideoCanvas(QWidget):
    """
    视频画面预览与 6 选区交互组件 (支持 1# 蓝绿橙 与 2# 紫青红 两套 ROI，支持数显及各选区旋转方向)
    """
    roi_changed_signal = pyqtSignal(str, tuple)  # roi_type, (x, y, w, h)
    roi_rotation_changed_signal = pyqtSignal(str, int)  # roi_type, angle (0, 90, 180, 270)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)

        self.current_frame = None       # QImage
        self._frame_buffer = None       # 保持底层 numpy 数组引用，防止被 GC 释放
        self.orig_w = 0
        self.orig_h = 0

        # 当前正在绘制的框: 'soc_1', 'time_1', 'cap_1', 'soc_2', 'time_2', 'cap_2'
        self.active_roi_type = 'soc_1'

        # 原始图像坐标下的 ROI 框 (支持双机 6 框)
        self.rois = {
            'soc_1': None,
            'time_1': None,
            'cap_1': None,
            'soc_2': None,
            'time_2': None,
            'cap_2': None
        }

        # 各选区旋转角度 (0: 正常, 90: 顺时针侧放, 180: 倒立, 270: 逆时针侧放)
        self.rotations = {
            'soc_1': 0,
            'time_1': 0,
            'cap_1': 0,
            'soc_2': 0,
            'time_2': 0,
            'cap_2': 0
        }

        # 选区颜色映射
        self.colors = {
            'soc_1': QColor(0, 150, 255),    # 蓝色 (1# 电量)
            'time_1': QColor(0, 200, 100),   # 绿色 (1# 时间)
            'cap_1': QColor(255, 140, 0),    # 橙色 (1# 容量)
            'soc_2': QColor(170, 85, 255),   # 紫色 (2# 电量)
            'time_2': QColor(0, 210, 210),   # 青色 (2# 时间)
            'cap_2': QColor(255, 75, 75)     # 红色 (2# 容量)
        }

        self.names = {
            'soc_1': '1# 充电宝电量',
            'time_1': '1# Power-Z 时间',
            'cap_1': '1# Power-Z 容量',
            'soc_2': '2# 充电宝电量',
            'time_2': '2# Power-Z 时间',
            'cap_2': '2# Power-Z 容量'
        }

        # 鼠标交互状态
        self.is_drawing = False
        self.start_pt = None
        self.current_pt = None

    def set_frame(self, cv_img):
        """设置当前帧图像 (OpenCV BGR numpy array)"""
        if cv_img is None:
            return
        self.orig_h, self.orig_w, ch = cv_img.shape
        bytes_per_line = ch * self.orig_w
        # BGR 转 RGB 并强制创建连续内存块
        rgb_img = cv_img[:, :, ::-1].copy()
        self._frame_buffer = rgb_img
        # 复制 QImage 避免底层内存指针释放后闪退
        q_img = QImage(rgb_img.data, self.orig_w, self.orig_h, bytes_per_line, QImage.Format_RGB888).copy()
        self.current_frame = q_img
        self.update()

    def set_active_roi_type(self, roi_type):
        aliases = {'soc': 'soc_1', 'time': 'time_1', 'cap': 'cap_1'}
        self.active_roi_type = aliases.get(roi_type, roi_type)

    def set_roi_rotation(self, roi_type, angle):
        """设置指定选区的旋转角度 (0, 90, 180, 270)"""
        aliases = {'soc': 'soc_1', 'time': 'time_1', 'cap': 'cap_1'}
        key = aliases.get(roi_type, roi_type)
        if key in self.rotations:
            self.rotations[key] = int(angle) % 360
            self.roi_rotation_changed_signal.emit(key, self.rotations[key])
            self.update()

    def get_roi_rotation(self, roi_type):
        aliases = {'soc': 'soc_1', 'time': 'time_1', 'cap': 'cap_1'}
        key = aliases.get(roi_type, roi_type)
        return self.rotations.get(key, 0)

    def rotate_active_roi(self, step=90):
        """快捷顺时针旋转当前激活选区 90 度"""
        if self.active_roi_type in self.rotations:
            new_ang = (self.rotations[self.active_roi_type] + step) % 360
            self.rotations[self.active_roi_type] = new_ang
            self.roi_rotation_changed_signal.emit(self.active_roi_type, new_ang)
            self.update()
            return self.active_roi_type, new_ang
        return None, 0

    def clear_roi(self, roi_type):
        aliases = {'soc': 'soc_1', 'time': 'time_1', 'cap': 'cap_1'}
        key = aliases.get(roi_type, roi_type)
        if key in self.rois:
            self.rois[key] = None
            self.update()

    def clear_unit_rois(self, unit=1):
        for k in [f'soc_{unit}', f'time_{unit}', f'cap_{unit}']:
            if k in self.rois:
                self.rois[k] = None
        self.update()

    def clear_all_rois(self):
        for k in self.rois:
            self.rois[k] = None
        self.update()

    def get_render_rect(self):
        """计算保持比例缩放后的绘制矩形区域"""
        if not self.current_frame:
            return QRect(0, 0, self.width(), self.height())

        widget_w = self.width()
        widget_h = self.height()
        aspect = self.orig_w / max(self.orig_h, 1)
        w_aspect = widget_w / max(widget_h, 1)

        if w_aspect > aspect:
            render_h = widget_h
            render_w = int(render_h * aspect)
            offset_x = (widget_w - render_w) // 2
            offset_y = 0
        else:
            render_w = widget_w
            render_h = int(render_w / aspect)
            offset_x = 0
            offset_y = (widget_h - render_h) // 2

        return QRect(offset_x, offset_y, render_w, render_h)

    def widget_to_orig_coord(self, pt):
        """将控件像素坐标转换为原图坐标"""
        if not self.current_frame or self.orig_w == 0:
            return (0, 0)
        r = self.get_render_rect()
        if r.width() == 0 or r.height() == 0:
            return (0, 0)

        norm_x = (pt.x() - r.x()) / r.width()
        norm_y = (pt.y() - r.y()) / r.height()

        orig_x = int(max(0, min(1.0, norm_x)) * self.orig_w)
        orig_y = int(max(0, min(1.0, norm_y)) * self.orig_h)
        return (orig_x, orig_y)

    def orig_to_widget_rect(self, roi):
        """将原图 ROI 坐标 (x, y, w, h) 转换为控件画布 QRect"""
        if not roi or not self.current_frame or self.orig_w == 0:
            return QRect()
        r = self.get_render_rect()
        x, y, w, h = roi
        wx = r.x() + int((x / self.orig_w) * r.width())
        wy = r.y() + int((y / self.orig_h) * r.height())
        ww = int((w / self.orig_w) * r.width())
        wh = int((h / self.orig_h) * r.height())
        return QRect(wx, wy, ww, wh)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.active_roi_type and self.current_frame:
            self.is_drawing = True
            self.start_pt = event.pos()
            self.current_pt = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.current_pt = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.current_pt = event.pos()

            # 计算原图选区
            x1, y1 = self.widget_to_orig_coord(self.start_pt)
            x2, y2 = self.widget_to_orig_coord(self.current_pt)

            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)

            if rw > 5 and rh > 5:
                roi = (rx, ry, rw, rh)
                self.rois[self.active_roi_type] = roi
                self.roi_changed_signal.emit(self.active_roi_type, roi)

            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 绘制背景
        painter.fillRect(self.rect(), QColor(25, 25, 30))

        # 2. 绘制视频帧
        if self.current_frame:
            r = self.get_render_rect()
            scaled_pixmap = QPixmap.fromImage(self.current_frame).scaled(r.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(r.x(), r.y(), scaled_pixmap)
        else:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Microsoft YaHei", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "请点击左侧【打开录屏视频文件】加载画面")
            return

        # 3. 绘制已标注的 6 套 ROIs
        font = QFont("Microsoft YaHei", 9, QFont.Bold)
        painter.setFont(font)

        for roi_type, roi in self.rois.items():
            if roi:
                qrect = self.orig_to_widget_rect(roi)
                color = self.colors.get(roi_type, QColor(255, 255, 255))
                rot = self.rotations.get(roi_type, 0)

                # 边框
                pen = QPen(color, 2, Qt.SolidLine)
                painter.setPen(pen)
                # 半透明填充
                fill_color = QColor(color.red(), color.green(), color.blue(), 40)
                painter.fillRect(qrect, fill_color)
                painter.drawRect(qrect)

                # 绘制文字顶部朝向指示条 (粗线指示哪边为文字正上方)
                dir_pen = QPen(QColor(255, 255, 0), 4, Qt.SolidLine)
                painter.setPen(dir_pen)
                if rot == 0:
                    painter.drawLine(qrect.topLeft(), qrect.topRight())
                elif rot == 90:
                    painter.drawLine(qrect.topRight(), qrect.bottomRight())
                elif rot == 180:
                    painter.drawLine(qrect.bottomLeft(), qrect.bottomRight())
                elif rot == 270:
                    painter.drawLine(qrect.topLeft(), qrect.bottomLeft())

                # 标签角标 (带旋转朝向说明)
                rot_text = ""
                if rot == 90:
                    rot_text = " ↻90°侧放"
                elif rot == 180:
                    rot_text = " ↻180°倒立"
                elif rot == 270:
                    rot_text = " ↺90°侧放"

                label = f"[{self.names.get(roi_type, roi_type)}{rot_text}]"
                label_w = len(label) * 11 + 16
                label_rect = QRect(qrect.x(), max(0, qrect.y() - 24), label_w, 22)
                painter.fillRect(label_rect, color)
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(font)
                painter.drawText(label_rect, Qt.AlignCenter, label)

        # 4. 绘制当前拖拽中的框
        if self.is_drawing and self.start_pt and self.current_pt:
            color = self.colors.get(self.active_roi_type, QColor(255, 255, 255))
            pen = QPen(color, 2, Qt.DashLine)
            painter.setPen(pen)

            dx = self.current_pt.x() - self.start_pt.x()
            dy = self.current_pt.y() - self.start_pt.y()
            rx = self.start_pt.x() if dx >= 0 else self.current_pt.x()
            ry = self.start_pt.y() if dy >= 0 else self.current_pt.y()
            drag_rect = QRect(rx, ry, abs(dx), abs(dy))

            painter.drawRect(drag_rect)

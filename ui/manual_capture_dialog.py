"""Per-unit, exact-frame manual completion of missing SOC records."""
import math
import os
import tempfile
from pathlib import Path
from datetime import time
from uuid import uuid4

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QDoubleSpinBox, QListWidget, QListWidgetItem, QLineEdit,
    QMessageBox, QProgressBar, QScrollArea, QWidget, QCheckBox)
from core.manual_capture import CaptureTask
from core.segment_dataset import read_image
from core.roi_geometry import crop_roi
from core.record_evidence import save_evidence


def pixmap(image):
    import cv2
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                   QImage.Format_RGB888).copy())


class ManualCaptureDialog(QDialog):
    def __init__(self, source, unit, soc, rois, rotations, start, end, duration,
                 cap_unit='Ah', parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'{unit}# 缺失电量手动补采 — 确认 {soc}%')
        self.resize(1000, 760)
        self.source, self.unit, self.soc = source, unit, soc
        self.rois, self.rotations, self.cap_unit = dict(rois), dict(rotations), cap_unit
        self.temp = tempfile.TemporaryDirectory(prefix='powerbank-candidates-')
        self.task = None
        self.selected = None
        self.fields = None
        self.record = None
        self.closing = False
        self.select_last = False
        layout = QVBoxLayout(self)
        hint = QLabel(f'{unit}# · 补采目标 {soc}%：选择正确图片，再读取同一帧的时间与容量。\n'
                      '选中的时刻不自动等于首次出现；可缩小范围、加密采集或逐帧向前核对。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        bar = QHBoxLayout()
        self.start_spin, self.end_spin, self.step_spin = (QDoubleSpinBox() for _ in range(3))
        for spin in (self.start_spin, self.end_spin):
            spin.setRange(0, max(0, duration))
            spin.setDecimals(3)
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)
        self.step_spin.setRange(.001, max(1, duration))
        self.step_spin.setDecimals(3)
        self.step_spin.setValue(max(.5, math.ceil(max(0, end-start) / 239 * 1000) / 1000))
        for name, spin in [('起点(秒)', self.start_spin), ('终点(秒)', self.end_spin), ('间隔(秒)', self.step_spin)]:
            bar.addWidget(QLabel(name)); bar.addWidget(spin)
        self.scan_button = QPushButton('采集候选图片')
        self.scan_button.clicked.connect(self.scan)
        bar.addWidget(self.scan_button)
        layout.addLayout(bar)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.candidates = QListWidget()
        self.candidates.setViewMode(QListWidget.IconMode)
        self.candidates.setResizeMode(QListWidget.Adjust)
        self.candidates.setIconSize(QSize(130, 75))
        metrics = self.candidates.fontMetrics()
        grid_height = 75 + metrics.height() * 2 + 24
        self.candidates.setGridSize(QSize(max(170, metrics.horizontalAdvance('11500.000s') + 30), grid_height))
        self.candidates.setMinimumHeight(grid_height + 15)
        self.candidates.setTextElideMode(Qt.ElideNone)
        self.candidates.currentItemChanged.connect(self.select_candidate)
        layout.addWidget(self.candidates, 2)
        self.preview = QLabel('选择候选图片后显示同帧电量、时间和容量截图')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(130)
        layout.addWidget(self.preview)
        steps = QHBoxLayout()
        self.previous_button, self.next_button = QPushButton('前一帧'), QPushButton('后一帧')
        self.previous_button.clicked.connect(lambda: self.step_frame(-1))
        self.next_button.clicked.connect(lambda: self.step_frame(1))
        self.read_button = QPushButton('读取选中帧的时间和容量')
        self.read_button.clicked.connect(self.read_selected)
        self.full_button = QPushButton('放大原始帧')
        self.full_button.clicked.connect(self.show_full_frame)
        for button in (self.previous_button, self.next_button, self.full_button, self.read_button):
            steps.addWidget(button)
        layout.addLayout(steps)
        self.single_line = QCheckBox('时间、容量框各只含一行数字（直接识别小数点和冒号）')
        self.single_line.toggled.connect(self.invalidate_fields)
        layout.addWidget(self.single_line)
        form = QFormLayout()
        self.time_edit, self.capacity_edit = QLineEdit(), QLineEdit()
        self.time_edit.setPlaceholderText('HH:MM:SS；无法辨认时留空')
        self.capacity_edit.setPlaceholderText('无法辨认时留空；可根据截图手动核对')
        form.addRow('测试仪时间', self.time_edit)
        form.addRow('容量（Ah）', self.capacity_edit)
        layout.addLayout(form)
        self.status = QLabel('尚未采集；每批最多 240 张，可分批加密。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.confirm_button = QPushButton(f'确认图片为 {soc}% 并补入 {unit}#')
        self.confirm_button.clicked.connect(self.confirm)
        self.cancel_button = QPushButton('取消 / 停止采集')
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.confirm_button); buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.set_busy(False)

    def set_busy(self, busy):
        for widget in (self.scan_button, self.start_spin, self.end_spin, self.step_spin, self.candidates):
            widget.setEnabled(not busy)
        for button in (self.previous_button, self.next_button, self.read_button, self.full_button):
            button.setEnabled(not busy and self.selected is not None)
        self.confirm_button.setEnabled(not busy and self.fields is not None)
        self.single_line.setEnabled(not busy)

    def invalidate_fields(self):
        self.fields = None
        self.confirm_button.setEnabled(False)
        self.time_edit.clear(); self.capacity_edit.clear()

    def launch(self, **kwargs):
        if self.task and self.task.isRunning():
            return
        self.set_busy(True)
        self.task = CaptureTask(self.source, self.temp.name, self.rois, self.rotations,
                                cap_unit=self.cap_unit, single_line=self.single_line.isChecked(), parent=self, **kwargs)
        self.task.candidate.connect(self.add_candidate)
        self.task.result.connect(self.read_complete)
        self.task.error.connect(self.on_error)
        self.task.progress.connect(self.progress.setValue)
        self.task.finished.connect(self.task_finished)
        self.task.start()

    def on_error(self, message):
        self.select_last = False
        self.status.setText('失败：' + message)

    def task_finished(self):
        if self.select_last and not self.closing:
            self.select_last = False
            self.candidates.setCurrentRow(self.candidates.count() - 1)
        self.set_busy(False)
        if self.closing:
            self.temp.cleanup()
            super().reject()

    def scan(self):
        if self.task and self.task.isRunning():
            return
        self.selected = self.fields = None
        self.candidates.clear()
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory(prefix='powerbank-candidates-')
        self.time_edit.clear(); self.capacity_edit.clear()
        self.status.setText('正在采集，完成后选择图片；可取消。')
        self.launch(start=self.start_spin.value(), end=self.end_spin.value(), step=self.step_spin.value())

    def add_candidate(self, entry):
        thumbnail = entry.pop('thumbnail')
        item = QListWidgetItem(QIcon(pixmap(thumbnail)), f"{entry['video_sec']:.3f}s\n帧 {entry['frame_index']}")
        item.setData(Qt.UserRole, entry)
        self.candidates.addItem(item)
        self.status.setText(f'已采集 {self.candidates.count()} 张；选择正确图片后读取同帧数据。')

    def select_candidate(self, current, previous=None):
        self.selected = current.data(Qt.UserRole) if current else None
        self.fields = None
        self.time_edit.clear(); self.capacity_edit.clear()
        if self.selected:
            frame = read_image(self.selected['path'])
            from PIL import Image, ImageDraw
            import cv2
            import numpy as np
            strip = Image.new('RGB', (900, 120), 'white')
            draw = ImageDraw.Draw(strip)
            for i, key in enumerate(('soc', 'time', 'cap')):
                crop = crop_roi(frame, self.rois[key], self.rotations.get(key, 0), strict=True)
                im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                im.thumbnail((285, 95))
                scale = min(285 / im.width, 95 / im.height)
                im = im.resize((max(1, int(im.width*scale)), max(1, int(im.height*scale))))
                strip.paste(im, (i*300, 22)); draw.text((i*300, 2), key, fill='black')
            self.preview.setPixmap(pixmap(cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)).scaled(
                900, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.status.setText(f"选中 {self.selected['video_sec']:.3f}s，帧 {self.selected['frame_index']}；尚未读取。")
        self.set_busy(bool(self.task and self.task.isRunning()))

    def read_selected(self):
        if self.selected:
            self.fields = None
            self.status.setText('正在读取选中原始帧，结果需对照截图核对。')
            self.launch(selected=dict(self.selected))

    def read_complete(self, fields):
        self.fields = fields
        self.time_edit.setText(fields['time_str'] if fields.get('time_obj') is not None else '')
        self.capacity_edit.setText(str(fields['capacity']) if fields['capacity'] is not None else '')
        self.status.setText(f"同帧读取完成：容量原文 {fields['capacity_raw']!r}。请核对截图；空白字段保留待核对。")

    def step_frame(self, direction):
        if not self.selected:
            return
        self.fields = None
        self.select_last = True
        self.launch(target_index=self.selected['frame_index'] + direction)

    def show_full_frame(self):
        if not self.selected:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.unit}# 原始帧 {self.selected['frame_index']} · {self.selected['video_sec']:.3f}s")
        dialog.resize(1000, 700)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        label = QLabel(); label.setPixmap(pixmap(read_image(self.selected['path'])))
        scroll.setWidget(label); layout.addWidget(scroll)
        dialog.exec_()

    def confirm(self):
        if self.fields is None or self.selected is None:
            return
        try:
            text = self.time_edit.text().strip()
            obj = time.fromisoformat(text) if text else None
            if obj and (obj.tzinfo is not None or obj.microsecond):
                raise ValueError('时间请使用 HH:MM:SS')
            cap_text = self.capacity_edit.text().strip()
            capacity = float(cap_text) if cap_text else None
            if capacity is not None and (not math.isfinite(capacity) or capacity < 0):
                raise ValueError('容量必须是非负有限数值')
        except ValueError as exc:
            QMessageBox.warning(self, '请检查输入', str(exc)); return
        if obj is None or capacity is None:
            if QMessageBox.question(self, '保留待核对', '时间或容量为空，将保留缺失状态，仍补入该帧？') != QMessageBox.Yes:
                return
        frame = read_image(self.selected['path'])
        directory = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'PowerbankTestTool/record_evidence' / uuid4().hex
        self.record = dict(soc=self.soc, time_str=obj.strftime('%H:%M:%S') if obj else '--',
                           time_obj=obj, capacity=capacity, is_target=True,
                           capture_method='manual', capture_unit=self.unit,
                           manual_override=(text != self.fields['time_str'] or capacity != self.fields['capacity']),
                           capacity_raw=self.fields['capacity_raw'])
        self.record.update(save_evidence(frame, self.source, self.selected['video_sec'],
                                         self.selected['frame_index'], directory))
        self.temp.cleanup()
        self.accept()

    def reject(self):
        if self.task and self.task.isRunning():
            self.closing = True
            self.task.requestInterruption()
            self.status.setText('正在停止后台任务…')
            return
        self.temp.cleanup()
        super().reject()

    def closeEvent(self, event):
        event.ignore()
        self.reject()

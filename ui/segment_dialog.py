"""Video collection and manual labelling, reachable only in segment mode."""
from pathlib import Path
from datetime import datetime
import math

import cv2
from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QFileDialog, QMessageBox,
    QTabWidget, QWidget, QListWidget, QListWidgetItem, QProgressBar, QSizePolicy,
    QInputDialog, QCheckBox, QApplication, QScrollArea, QFrame)

from core.segment_dataset import (data_home, read_json, write_json, read_image,
    samples, set_label, SampleCaptureWorker, capture_sessions, library_paths,
    template_catalog, register_template, DatasetImportWorker)
from core.segment_dataset import (export_template_package, import_template_package,
    clean_all_unlabelled_samples, load_ignore_regions, save_ignore_regions, crop_digits)
from ui.ignore_regions_dialog import IgnoreRegionsDialog
from core.roi_geometry import roi_in_bounds
from ui.video_canvas import VideoCanvas
from ui.input_guard import disable_wheel_changes


class SegmentDatasetDialog(QDialog):
    def __init__(self, paths, current_video=None, rois=None, rotations=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('数码管专用 · 视频采集与样本标注')
        self.paths = {unit: library_paths(paths.get(unit)) for unit in (1, 2)}
        self.settings = QSettings('PowerbankTestTool', 'SegmentCapture')
        self.edit_root = next(iter(self.paths[1] or self.paths[2]), str(data_home() / 'unit_1'))
        self.loaded_root = None
        self._labels = {}
        self._all_rows = []
        self.cap = None
        self._capture_frame_valid = False
        self.worker = None
        self._pending_close_result = None
        self.rows = []
        self.current_video = current_video
        self.apply_rois = None
        self.apply_rotations = None
        self.initial_rois, self.initial_rotations = rois or {}, rotations or {}
        # 内容区允许纵向滚动，避免低分辨率或高 DPI 缩放时底部操作被挤出屏幕。
        # 状态和保存按钮留在滚动区外，确保用户随时可以完成或退出设置。
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(6, 6, 6, 6)
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QFrame.NoFrame)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        content_scroll.setWidget(content)
        outer_layout.addWidget(content_scroll, 1)
        top = QHBoxLayout()
        self.unit = QComboBox()
        self.unit.addItems(['采集画面：1#', '采集画面：2#'])
        top.addWidget(self.unit)
        self.library = QComboBox()
        self.library.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.library.setMinimumContentsLength(10)
        self.library.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        top.addWidget(self.library, 1)
        self.new_library = QPushButton('新建模板')
        self.new_library.clicked.connect(self.create_library)
        top.addWidget(self.new_library)
        self.rename_library = QPushButton('重命名')
        self.rename_library.clicked.connect(self.rename_current_library)
        top.addWidget(self.rename_library)
        self.choose_library = QPushButton('登记已有模板')
        self.choose_library.clicked.connect(self.pick_library)
        top.addWidget(self.choose_library)
        layout.addLayout(top)
        self.library_info = QLabel('')
        self.library_info.setWordWrap(True)
        self.library_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.library_info)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        collect = QWidget()
        cl = QVBoxLayout(collect)
        sources = QHBoxLayout()
        self.source = QComboBox()
        self.source.setMinimumWidth(180)
        self.source.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.source.setMinimumContentsLength(12)
        self.source.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        sources.addWidget(self.source, 1)
        self.add_files = QPushButton('添加视频')
        self.add_folder = QPushButton('添加文件夹内视频')
        self.add_files.clicked.connect(self.pick_files)
        self.add_folder.clicked.connect(self.pick_folder)
        sources.addWidget(self.add_files)
        sources.addWidget(self.add_folder)
        cl.addLayout(sources)
        tip = QLabel('拖框只保留完整电量数字；拖动框上圆形手柄可自由旋转，双击恢复0°。不同外观请使用不同样本库。')
        tip.setWordWrap(True)
        cl.addWidget(tip)
        self.canvas = VideoCanvas()
        self.canvas.setMinimumSize(320, 180)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cl.addWidget(self.canvas, 1)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderReleased.connect(self.preview)
        cl.addWidget(self.slider)
        self.frame_info = QLabel('选择视频后，拖动滑块并松开预览。')
        cl.addWidget(self.frame_info)
        self.edit_capture_ignore = QPushButton('设置当前模板忽略区域（闪电等图标）')
        self.edit_capture_ignore.clicked.connect(lambda: self.edit_ignore_regions(False))
        cl.addWidget(self.edit_capture_ignore)
        controls = QHBoxLayout()
        self.angle = QComboBox()
        self.angle.addItems(['正向 0°', '顺时针侧放 90°', '倒立 180°', '逆时针侧放 270°'])
        self.angle.currentIndexChanged.connect(self.change_angle)
        controls.addWidget(self.angle)
        self.start_sec, self.end_sec, self.interval = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for title, spin in [('开始秒', self.start_sec), ('结束秒', self.end_sec), ('间隔秒', self.interval)]:
            spin.setRange(0, 999999)
            spin.setDecimals(1)
            controls.addWidget(QLabel(title))
            controls.addWidget(spin)
        self.interval.setRange(.2, 3600)
        self.interval.setValue(float(self.settings.value('interval', 5)))
        self.interval.setSingleStep(.1)
        self.interval.setKeyboardTracking(False)
        self.interval.valueChanged.connect(lambda value: self.settings.setValue('interval', value))
        cl.addLayout(controls)
        self.capture_policy = QComboBox()
        self.capture_policy.addItems(['追加一个新批次', '重新采集（隐藏同视频同位置的旧批次，保留图片与标注）'])
        cl.addWidget(self.capture_policy)
        self.capture_summary = QLabel()
        cl.addWidget(self.capture_summary)
        for spin in (self.start_sec, self.end_sec, self.interval):
            spin.valueChanged.connect(self.update_capture_summary)
        self.update_capture_summary()
        buttons = QHBoxLayout()
        self.capture = QPushButton('开始采集当前视频的数字截图')
        self.cancel = QPushButton('停止采集（保留已采集图片）')
        self.cancel.setEnabled(False)
        self.capture.clicked.connect(self.start_capture)
        self.cancel.clicked.connect(self.stop_capture)
        buttons.addWidget(self.capture)
        buttons.addWidget(self.cancel)
        cl.addLayout(buttons)
        self.progress = QProgressBar()
        cl.addWidget(self.progress)
        self.tabs.addTab(collect, '1. 视频采集')

        review = QWidget()
        rl = QVBoxLayout(review)
        rl.addWidget(QLabel('只有人工确认的数字才参与匹配。缺笔画、闪烁、遮挡请选择“不可辨认”，不要猜测。'))
        batch_row = QHBoxLayout()
        self.batch = QComboBox()
        self.batch.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.batch.setMinimumContentsLength(28)
        self.batch.currentIndexChanged.connect(self.reset_page)
        batch_row.addWidget(self.batch, 1)
        self.show_history = QCheckBox('显示隐藏批次')
        self.show_history.toggled.connect(self.refresh_samples)
        batch_row.addWidget(self.show_history)
        self.restore_batch = QPushButton('恢复本批次')
        self.restore_batch.clicked.connect(self.restore_current_batch)
        batch_row.addWidget(self.restore_batch)
        self.import_samples = QPushButton('导入数据集到当前模板')
        self.import_samples.clicked.connect(self.import_to_library)
        batch_row.addWidget(self.import_samples)
        rl.addLayout(batch_row)
        self.filter = QComboBox()
        self.filter.addItems(['全部样本', '未标注', '已标注数字', '不可辨认'])
        self.filter.currentIndexChanged.connect(self.reset_page)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.filter, 1)
        filter_row.addWidget(QLabel('页码：'))
        self.page = QSpinBox()
        self.page.setRange(1, 1)
        self.page.valueChanged.connect(self.render_samples)
        filter_row.addWidget(self.page)
        self.page_info = QLabel()
        filter_row.addWidget(self.page_info)
        self.clear_unlabelled = QPushButton('清空当前模板未标注样本')
        self.clear_unlabelled.clicked.connect(self.clear_all_unlabelled)
        filter_row.addWidget(self.clear_unlabelled)
        rl.addLayout(filter_row)
        body = QHBoxLayout()
        self.list = QListWidget()
        self.list.setMaximumWidth(470)
        self.list.currentRowChanged.connect(self.show_sample)
        body.addWidget(self.list)
        right = QVBoxLayout()
        self.picture = QLabel('选择样本查看')
        self.picture.setAlignment(Qt.AlignCenter)
        self.picture.setMinimumSize(240, 180)
        right.addWidget(self.picture, 1)
        self.edit_sample_ignore = QPushButton('用这张样本设置模板忽略区域')
        self.edit_sample_ignore.clicked.connect(lambda: self.edit_ignore_regions(True))
        right.addWidget(self.edit_sample_ignore)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        right.addWidget(self.details)
        self.label_value = QSpinBox()
        self.label_value.setRange(0, 100)
        self.label_value.setSuffix('%')
        right.addWidget(self.label_value)
        for title, action in [
            ('保存标签并查看下一张', lambda: self.label_sample(self.label_value.value())),
            ('不可辨认 / 数字未完整显示', lambda: self.label_sample('invalid')),
            ('撤销此图片的标注', lambda: self.label_sample(None)),
        ]:
            button = QPushButton(title)
            button.clicked.connect(action)
            right.addWidget(button)
        body.addLayout(right, 1)
        rl.addLayout(body, 1)
        self.coverage = QLabel()
        self.coverage.setWordWrap(True)
        rl.addWidget(self.coverage)
        self.tabs.addTab(review, '2. 标注与样本库')
        templates_page = QWidget()
        tl = QVBoxLayout(templates_page)
        intro = QLabel('模板可以重复用于任何样机。分别勾选 1#、2# 要使用的模板，可单选、多选或全选。\n采集位置只决定截取哪个框，不会改变下面的模板选择。已有标签原样保留。')
        intro.setWordWrap(True)
        tl.addWidget(intro)
        share_row = QHBoxLayout()
        self.export_template = QPushButton('导出当前模板包（可分享）')
        self.export_template.clicked.connect(self.export_current_template)
        self.import_template = QPushButton('导入别人发来的模板包')
        self.import_template.clicked.connect(self.import_shared_template)
        share_row.addWidget(self.export_template)
        share_row.addWidget(self.import_template)
        tl.addLayout(share_row)
        selections = QHBoxLayout()
        self.template_checks = {}
        for unit in (1, 2):
            column = QVBoxLayout()
            column.addWidget(QLabel(f'{unit}# 分析时使用的模板'))
            check_list = QListWidget()
            self.template_checks[unit] = check_list
            column.addWidget(check_list)
            row = QHBoxLayout()
            for title, checked in [('全选', True), ('全不选', False)]:
                button = QPushButton(title)
                button.clicked.connect(lambda _, u=unit, c=checked: self.select_all_templates(u, c))
                row.addWidget(button)
            column.addLayout(row)
            selections.addLayout(column)
            check_list.itemChanged.connect(self.sync_template_checks)
        tl.addLayout(selections)
        self.tabs.addTab(templates_page, '3. 测量模板选择')
        self.status = QLabel('截图只在本机保存；采集不会自动给图片贴上 OCR 标签。')
        self.status.setWordWrap(True)
        outer_layout.addWidget(self.status)
        self.use_selection = QPushButton('保存选择，并将当前视频的采集框用于分析')
        self.use_selection.clicked.connect(self.use_current_selection)
        outer_layout.addWidget(self.use_selection)
        self.save_selection = QPushButton('保存模板选择并返回主界面（保持原选区）')
        self.save_selection.clicked.connect(self.accept)
        outer_layout.addWidget(self.save_selection)
        self.unit.currentIndexChanged.connect(self.change_unit)
        self.source.currentIndexChanged.connect(self.open_source)
        self.library.currentIndexChanged.connect(self.change_library)
        self.reload_catalog()
        self.change_unit()
        saved = read_json(data_home() / 'sources.json', [])
        self.add_sources(([current_video] if current_video else []) + saved)
        self._wheel_guard = disable_wheel_changes(self)

        # 以当前显示器可用区域为上限。高 DPI 下 Qt 返回的是逻辑像素，
        # 因此这里同样适用于 125%/150% 的 Windows 显示缩放。
        screen = parent.screen() if parent is not None and parent.screen() else QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(1080, max(1, int(available.width() * 0.94))),
                        min(820, max(1, int(available.height() * 0.92))))
        else:
            self.resize(1080, 820)

    def unit_id(self):
        return self.unit.currentIndex() + 1

    def root(self):
        return Path(self.edit_root)

    def reload_catalog(self):
        entries = template_catalog([self.edit_root] + self.paths[1] + self.paths[2])
        self.library.blockSignals(True)
        self.library.clear()
        for entry in entries:
            self.library.addItem(entry['name'], entry['path'])
        selected = self.library.findData(self.edit_root)
        self.library.setCurrentIndex(max(0, selected))
        self.library.blockSignals(False)
        for unit, widget in self.template_checks.items():
            widget.blockSignals(True)
            widget.clear()
            selected_paths = {str(Path(p).resolve()).casefold() for p in self.paths[unit]}
            for entry in entries:
                labels = read_json(Path(entry['path']) / 'labels.json', {})
                count = len({v for v in labels.values() if type(v) is int})
                item = QListWidgetItem(f'{entry["name"]}（已标 {count} 种电量）')
                item.setData(Qt.UserRole, entry['path'])
                item.setToolTip(entry['path'])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if str(Path(entry['path']).resolve()).casefold() in selected_paths else Qt.Unchecked)
                widget.addItem(item)
            widget.blockSignals(False)
        self.change_library()

    def sync_template_checks(self):
        for unit, widget in self.template_checks.items():
            self.paths[unit] = [widget.item(i).data(Qt.UserRole) for i in range(widget.count())
                                if widget.item(i).checkState() == Qt.Checked]

    def select_all_templates(self, unit, checked):
        widget = self.template_checks[unit]
        widget.blockSignals(True)
        for i in range(widget.count()):
            widget.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        widget.blockSignals(False)
        self.sync_template_checks()

    def create_library(self):
        name, ok = QInputDialog.getText(self, '新建样本集模板', '模板名称（例如 PB516 白色数码管）：')
        if ok and name.strip():
            self.edit_root = register_template(name)
            self.reload_catalog()
            self.status.setText('模板已保存。现在可以采集视频或导入已有数据集；到第三页勾选用于分析。')

    def rename_current_library(self):
        name, ok = QInputDialog.getText(self, '重命名模板', '模板名称：', text=self.library.currentText())
        if ok and name.strip():
            register_template(name, self.root())
            self.reload_catalog()

    def change_library(self):
        self.edit_root = self.library.currentData() or self.edit_root
        self.update_ignore_summary()
        self.refresh_samples()

    def update_ignore_summary(self):
        try:
            count = len(load_ignore_regions(self.root()))
            summary = f' · 忽略区域：{count} 个'
        except ValueError:
            summary = ' · 忽略区域配置无法读取'
        self.library_info.setText('当前编辑模板：' + str(self.root()) + summary)

    def edit_ignore_regions(self, from_sample=False):
        try:
            if from_sample:
                index = self.list.currentRow()
                if not 0 <= index < len(self.rows):
                    raise ValueError('请先选择一张数字完整的样本图片。')
                image = read_image(self.root() / self.rows[index]['key'])
            else:
                roi = self.canvas.rois.get(f'soc_{self.unit_id()}')
                if (not self._capture_frame_valid or self.cap is None or not self.cap.isOpened()
                        or self.canvas._frame_buffer is None or not roi):
                    raise ValueError('请先预览视频，并框选完整的电量数字区域。')
                frame = cv2.cvtColor(self.canvas._frame_buffer, cv2.COLOR_RGB2BGR)
                image = crop_digits(frame, roi, self.canvas.rotations[f'soc_{self.unit_id()}'])
            if image is None or not image.size:
                raise ValueError('无法读取数字图片，请重新选择。')
            dialog = IgnoreRegionsDialog(image, load_ignore_regions(self.root()), self)
            if dialog.exec_() == QDialog.Accepted:
                save_ignore_regions(self.root(), dialog.canvas.regions)
                self.update_ignore_summary()
                self.status.setText('忽略区域已保存到当前模板；已有与新采集样本都会使用，原图片和标注保留。')
        except (ValueError, OSError, cv2.error) as exc:
            QMessageBox.warning(self, '忽略区域设置失败', str(exc))

    def update_capture_summary(self):
        count = max(0, math.ceil((self.end_sec.value()-self.start_sec.value())/self.interval.value()))
        self.capture_summary.setText(f'本次采集间隔 {self.interval.value():g} 秒，预计 {count} 张。此设置不改变主界面的分析抽帧间隔。')

    def import_to_library(self):
        folder = QFileDialog.getExistingDirectory(self, '导入含 captures 和 labels.json 的数据集')
        if not folder:
            return
        self.worker = DatasetImportWorker(folder, str(self.root()))
        self.worker.completed.connect(lambda n: self.capture_done(str(self.root()), n, False))
        self.worker.failed.connect(self.capture_error)
        self.worker.finished.connect(lambda: self.set_busy(False))
        self.worker.finished.connect(self._finish_pending_close)
        self.set_busy(True)
        self.cancel.setEnabled(False)
        self.status.setText('正在复制数据集及标注到当前模板；原数据保留。')
        self.worker.start()

    def export_current_template(self):
        default = str(self.root().parent / (self.library.currentText() + '.zip'))
        path, _ = QFileDialog.getSaveFileName(self, '导出数码管模板包', default, '模板包 (*.zip)')
        if not path:
            return
        try:
            count = export_template_package(self.root(), path, self.library.currentText())
            QMessageBox.information(self, '导出完成', f'已打包 {count} 张已标注图片：\n{path}')
        except Exception as exc:
            QMessageBox.warning(self, '导出失败', str(exc))

    def import_shared_template(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入数码管模板包', '', '模板包 (*.zip)')
        if not path:
            return
        try:
            root, count = import_template_package(path)
            self.edit_root = root
            self.paths[self.unit_id()].append(root)
            self.reload_catalog()
            QMessageBox.information(self, '导入完成', f'已导入 {count} 张已标注图片，当前样机已勾选此模板。')
        except Exception as exc:
            QMessageBox.warning(self, '导入失败', str(exc))

    def add_sources(self, paths):
        existing = {self.source.itemData(i) for i in range(self.source.count())}
        for p in paths:
            p = str(Path(p))
            if p not in existing and Path(p).is_file():
                self.source.addItem(p, p)
                existing.add(p)
        write_json(data_home() / 'sources.json', list(existing))

    def pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '添加采集视频', '', '视频 (*.mp4 *.avi *.mkv *.mov)')
        self.add_sources(paths)

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择视频文件夹')
        if folder:
            self.add_sources([str(p) for p in Path(folder).iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mkv', '.mov'}])

    def pick_library(self):
        folder = QFileDialog.getExistingDirectory(self, '选择或新建独立样本库（不同设备外观请分开）', str(self.root().parent))
        if folder:
            self.edit_root = register_template(Path(folder).name, folder)
            self.reload_catalog()

    def change_unit(self):
        key = f'soc_{self.unit_id()}'
        self.canvas.set_active_roi_type(key)
        self.angle.setCurrentIndex(self.canvas.get_roi_rotation(key)//90)

    def change_angle(self, index):
        self.canvas.set_roi_rotation(f'soc_{self.unit_id()}', index*90)

    def open_source(self):
        self.clear_capture_preview()
        self.canvas.clear_all_rois()
        if self.cap:
            self.cap.release()
        self.cap = None
        path = self.source.currentData()
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, '无法读取', path)
            return
        fps, count = self.cap.get(cv2.CAP_PROP_FPS), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            self.cap.release()
            self.cap = None
            return
        self.slider.setRange(0, max(0, count-1))
        self.slider.setValue(0)
        self.end_sec.setValue(count/fps)
        self.start_sec.setValue(0)
        self.canvas.clear_all_rois()
        self.canvas.rotations = {k: 0 for k in self.canvas.rotations}
        presets = read_json(data_home() / 'presets.json', {}).get(str(Path(path)), {})
        for unit, preset in presets.items():
            key = f'soc_{unit}'
            if key in self.canvas.rois:
                self.canvas.rois[key] = tuple(preset['roi'])
                self.canvas.rotations[key] = preset['rotation']
        if self.current_video and Path(path) == Path(self.current_video):
            for key in ('soc_1', 'soc_2'):
                if self.initial_rois.get(key):
                    self.canvas.rois[key] = self.initial_rois[key]
                    self.canvas.rotations[key] = self.initial_rotations.get(key, 0)
        self.change_unit()
        self.preview()

    def preview(self):
        self.clear_capture_preview()
        if self.cap is None or not self.cap.isOpened():
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.slider.value())
        ok, frame = self.cap.read()
        if ok:
            self.canvas.set_frame(frame)
            self._capture_frame_valid = True
            sec = self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            self.frame_info.setText(f'帧 {self.slider.value()} · 视频时间 {sec:.2f} 秒')

    def clear_capture_preview(self):
        self._capture_frame_valid = False
        self.canvas.current_frame = None
        self.canvas._frame_buffer = None
        self.canvas.orig_w = self.canvas.orig_h = 0
        self.canvas.update()
        self.frame_info.setText('当前没有可用画面，请选择可读取的视频位置。')

    def set_busy(self, busy):
        for widget in (self.source, self.unit, self.choose_library, self.new_library, self.rename_library,
                       self.library, self.add_files, self.add_folder, self.capture_policy, self.save_selection,
                       self.slider, self.canvas, self.angle, self.start_sec, self.end_sec, self.interval,
                       self.capture, self.use_selection, self.export_template, self.import_template,
                       self.edit_capture_ignore, self.edit_sample_ignore):
            widget.setEnabled(not busy)
        self.tabs.setTabEnabled(1, not busy)
        self.tabs.setTabEnabled(2, not busy)
        self.cancel.setEnabled(busy)
        unlabelled = sum(row['key'] not in self._labels for row in self._all_rows)
        self.clear_unlabelled.setEnabled(not busy and unlabelled > 0)

    def start_capture(self):
        for spin in (self.interval, self.start_sec, self.end_sec):
            spin.interpretText()
        roi = self.canvas.rois[f'soc_{self.unit_id()}']
        if self.cap is None or not self.cap.isOpened() or not roi:
            QMessageBox.information(self, '先画框', '请选择视频，并在画面上框住完整的电量数字。')
            return
        if not roi_in_bounds(roi, self.canvas.orig_w, self.canvas.orig_h):
            QMessageBox.information(self, '选区超出边界', '请调整红色选区，确保旋转后的整个框都在视频画面内。')
            return
        if self.start_sec.value() >= self.end_sec.value():
            QMessageBox.information(self, '时间范围', '结束时间必须大于开始时间。')
            return
        presets = read_json(data_home() / 'presets.json', {})
        presets.setdefault(str(Path(self.source.currentData())), {})[str(self.unit_id())] = {
            'path': str(self.root()), 'roi': list(roi), 'rotation': self.angle.currentIndex()*90}
        write_json(data_home() / 'presets.json', presets)
        self.worker = SampleCaptureWorker(self.source.currentData(), self.root(), roi,
            self.angle.currentIndex()*90, self.unit_id(), self.interval.value(),
            self.start_sec.value(), self.end_sec.value())
        self.worker.progress.connect(lambda n, total: self.progress.setValue(int(n*100/max(total, 1))))
        self.worker.completed.connect(self.capture_done)
        self.worker.failed.connect(self.capture_error)
        self.worker.finished.connect(lambda: self.set_busy(False))
        self.worker.finished.connect(self._finish_pending_close)
        self.set_busy(True)
        self.progress.setValue(0)
        self.status.setText(f'正在采集到模板“{self.library.currentText()}”，每 {self.interval.value():g} 秒一张。')
        self.worker.start()

    def stop_capture(self):
        if self.worker:
            self.worker.stop()
            self.status.setText('正在停止并保存已采集的图片…')

    def capture_done(self, root, count, cancelled):
        batch_id = getattr(self.worker, 'batch_id', None)
        if batch_id and not cancelled and self.capture_policy.currentIndex() == 1:
            hidden = set(read_json(self.root() / 'hidden_batches.json', []))
            for session in capture_sessions(root):
                if (session['id'] != batch_id and session['source'] == self.source.currentData()
                        and session['unit'] == self.unit_id()):
                    hidden.add(session['id'])
            write_json(self.root() / 'hidden_batches.json', sorted(hidden))
        self.status.setText(f'{"已停止" if cancelled else "采集完成"}：保存 {count} 张到 {root}。请到第二页确认标签。')
        self.refresh_samples(preferred=batch_id, newest=True)

    def capture_error(self, message):
        self.status.setText('采集失败：' + message)
        self.refresh_samples(preferred=getattr(self.worker, 'batch_id', None), newest=True)

    def refresh_samples(self, *_, preferred=None, newest=False):
        try:
            self._labels = read_json(self.root() / 'labels.json', {})
            self._all_rows = samples(self.root())
            unlabelled_count = sum(row['key'] not in self._labels for row in self._all_rows)
            self.clear_unlabelled.setText(f'清空未标注样本（{unlabelled_count}张）')
            self.clear_unlabelled.setEnabled(unlabelled_count > 0 and not (
                self.worker and self.worker.isRunning()))
            old = preferred or (self.batch.currentData() if self.loaded_root == self.root() and not newest else None)
            self.loaded_root = self.root()
            hidden = set(read_json(self.root() / 'hidden_batches.json', []))
            self.batch.blockSignals(True)
            self.batch.clear()
            for session in capture_sessions(self.root()):
                if session['id'] in hidden and not self.show_history.isChecked():
                    continue
                stamp = datetime.fromtimestamp(session['created_at']).strftime('%m-%d %H:%M:%S')
                text = f'{stamp} · 间隔{session["interval"]:g}s · {len(session["samples"])}张 · {Path(session["source"]).name}'
                if session['id'] in hidden:
                    text = '[已隐藏] ' + text
                self.batch.addItem(text, session['id'])
            self.batch.addItem('全部可见批次', '*')
            index = self.batch.findData(old) if old else 0
            self.batch.setCurrentIndex(max(0, index))
            self.batch.blockSignals(False)
            if self.batch.currentData() != old:
                self.page.blockSignals(True)
                self.page.setValue(1)
                self.page.blockSignals(False)
            self.render_samples()
        except Exception as exc:
            QMessageBox.warning(self, '样本库读取失败', str(exc))

    def clear_all_unlabelled(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, '正在处理', '请先停止或等待采集、导入完成。')
            return
        labels = read_json(self.root() / 'labels.json', {})
        count = sum(row['key'] not in labels for row in samples(self.root()))
        if count == 0:
            QMessageBox.information(self, '没有未标注样本', '当前模板没有需要清理的未标注图片。')
            return
        if QMessageBox.question(self, '确认清空未标注样本',
                f'确定删除当前模板“{self.library.currentText()}”中的 {count} 张未标注图片吗？\n\n'
                '已标注数字和“不可辨认”样本都会保留。此操作不能撤销。',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            removed = clean_all_unlabelled_samples(self.root())
            self.refresh_samples()
            self.status.setText(f'已清理 {removed} 张未标注图片；已标注样本保持不变。')
        except Exception as exc:
            QMessageBox.warning(self, '清理失败', str(exc))

    def restore_current_batch(self):
        hidden = set(read_json(self.root() / 'hidden_batches.json', []))
        hidden.discard(self.batch.currentData())
        write_json(self.root() / 'hidden_batches.json', sorted(hidden))
        self.refresh_samples()

    def reset_page(self):
        self.page.blockSignals(True)
        self.page.setValue(1)
        self.page.blockSignals(False)
        self.render_samples()

    def render_samples(self):
        try:
            labels = self._labels
            hidden = set(read_json(self.root() / 'hidden_batches.json', []))
            batch_id = self.batch.currentData()
            all_rows = [r for r in self._all_rows if (r['batch'] not in hidden or self.show_history.isChecked())
                        and (batch_id == '*' or r['batch'] == batch_id)]
            selected = self.filter.currentIndex()
            filtered = [row for row in all_rows if selected == 0 or
                (selected == 1 and row['key'] not in labels) or
                (selected == 2 and type(labels.get(row['key'])) is int) or
                (selected == 3 and labels.get(row['key']) == 'invalid')]
            self.page.blockSignals(True)
            self.page.setRange(1, max(1, math.ceil(len(filtered)/500)))
            self.page.blockSignals(False)
            offset = (self.page.value()-1)*500
            self.rows = filtered[offset:offset+500]
            self.page_info.setText(f'/ {self.page.maximum()}（每页 500 张）')
            self.list.blockSignals(True)
            self.list.clear()
            for row in self.rows:
                label = labels.get(row['key'], '未标注')
                if label == 'invalid':
                    label = '不可辨认'
                self.list.addItem(f'{row["video_sec"]:.2f}s · {label} · {Path(row["source"]).name}')
            self.list.blockSignals(False)
            values = sorted({v for v in labels.values() if type(v) is int})
            self.coverage.setText(f'共 {len(all_rows)} 张，已确认电量：{", ".join(map(str, values)) or "无"}。尚无样本的电量不能保证识别。')
            self.list.setCurrentRow(0)
            if not self.rows:
                self.picture.clear()
                self.details.setText('当前筛选下没有样本。')
        except Exception as exc:
            QMessageBox.warning(self, '样本库读取失败', str(exc))

    def show_sample(self, index):
        if not 0 <= index < len(self.rows):
            return
        row = self.rows[index]
        image = read_image(self.root() / row['key'])
        if image is None:
            self.picture.clear()
            self.details.setText('图片无法读取')
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.picture.setPixmap(QPixmap.fromImage(qimage).scaled(480, 320, Qt.KeepAspectRatio, Qt.FastTransformation))
        label = self._labels.get(row['key'])
        if type(label) is int:
            self.label_value.setValue(label)
        elif label == 'invalid':
            label = '不可辨认'
        self.details.setText(f'视频：{row["source"]}\n时刻：{row["video_sec"]:.3f}s · 帧号：{row["frame_index"]}\n标签：{label if label is not None else "未标注"}')

    def label_sample(self, label):
        index = self.list.currentRow()
        if not 0 <= index < len(self.rows):
            return
        try:
            key = self.rows[index]['key']
            set_label(self.root(), key, label)
            if label is None:
                self._labels.pop(key, None)
            else:
                self._labels[key] = label
            self.render_samples()
            self.status.setText('已保存到当前模板。' if label is not None else '已保存：撤销了此标签，原图片保留。')
            # Filtered rows disappear after labelling; otherwise advance one row.
            step = 1 if self.filter.currentIndex() == 0 else 0
            if index+step >= len(self.rows) and self.page.value() < self.page.maximum():
                self.page.setValue(self.page.value()+1)
            else:
                self.list.setCurrentRow(min(index+step, len(self.rows)-1))
        except Exception as exc:
            QMessageBox.warning(self, '保存标签失败', str(exc))

    def use_current_selection(self):
        path = self.source.currentData()
        if not path or not self.current_video or Path(path) != Path(self.current_video):
            QMessageBox.information(self, '视频不同', '请先在主界面打开同一个视频，再使用它的采集选区。其他视频的样本已经保存在本地。')
            return
        self.apply_rois = {k: self.canvas.rois[k] for k in ('soc_1', 'soc_2')}
        self.apply_rotations = dict(self.canvas.rotations)
        self.accept()

    def done(self, result):
        if self.worker and self.worker.isRunning():
            self._pending_close_result = result
            self.stop_capture()
            self.status.setText('正在结束后台任务，完成后会自动关闭窗口。')
            return
        if self.cap:
            self.cap.release()
        super().done(result)

    def _finish_pending_close(self):
        if self._pending_close_result is None:
            return
        if self.worker and self.worker.isRunning():
            QTimer.singleShot(100, self._finish_pending_close)
            return
        result = self._pending_close_result
        self._pending_close_result = None
        self.done(result)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self._pending_close_result = QDialog.Rejected
            self.stop_capture()
            self.status.setText('正在结束后台任务，完成后会自动关闭窗口。')
            event.ignore()
        else:
            if self.cap:
                self.cap.release()
            event.accept()

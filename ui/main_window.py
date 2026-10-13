# 必须在最顶部导入 onnxruntime，避免 Windows 下 C++ 运行时 DLL 冲突
import onnxruntime
import os
import json
import re
import sys
import csv
import cv2
import traceback
from datetime import time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QRadioButton,
    QButtonGroup, QSlider, QDoubleSpinBox, QSpinBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox,
    QSplitter, QFrame, QScrollArea, QCheckBox, QInputDialog, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSettings, QTimer, QUrl
from PyQt5.QtGui import QFont, QIcon, QColor, QDesktopServices

from ui.video_canvas import VideoCanvas
from core.roi_geometry import crop_roi, roi_in_bounds
from core.record_evidence import LOCATOR_FIELDS, load_snapshot, locate_record
from core.segment_dataset import read_frame_at, mask_ocr_regions, validate_ignore_regions
from ui.ignore_regions_dialog import IgnoreRegionsDialog
from ui.input_guard import disable_wheel_changes
from core.video_worker import VideoAnalysisWorker
from core.excel_filler import (
    scan_excel_test_groups, get_target_soc_list
)
from core.ocr_engine import OCREngine
from core.segment_dataset import SEGMENT_MODE, SegmentMatcher, data_home, library_paths, template_catalog
from ui.segment_dialog import SegmentDatasetDialog
from core.project_archive import (ProjectArchive, create_project, delete_project,
    project_catalog, group_key)
from core.resource_monitor import ResourceMonitor
from core.performance import high_speed_threads


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("充电宝数显均匀度测试自动数据提取工具 v2.0 (支持双样机同步测试)")
        self.resize(1420, 900)

        # 核心数据
        self.video_path = None
        self.video_cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_cv = None
        self._preview_frame_valid = False

        template_name = 'PB519数显均匀度测试表(9.15修带标).xlsx'
        possible_paths = []
        import glob
        if getattr(sys, 'frozen', False):
            possible_paths.append(os.path.join(sys._MEIPASS, template_name))
            exe_dir = os.path.dirname(sys.executable)
            possible_paths.append(os.path.join(exe_dir, template_name))
            possible_paths.append(os.path.join(exe_dir, '_internal', template_name))
            possible_paths.extend(glob.glob(os.path.join(sys._MEIPASS, '*.xlsx')))
            possible_paths.extend(glob.glob(os.path.join(exe_dir, '_internal', '*.xlsx')))
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            possible_paths.append(os.path.join(base_dir, template_name))
            possible_paths.extend(glob.glob(os.path.join(base_dir, '*.xlsx')))
            
        found_path = ''
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
                
        self.excel_template_path = found_path if found_path else os.path.abspath(template_name)
        self.excel_groups = {}
        self.extracted_records_1 = {}  # {soc: {...}} 1# 样机记录
        self.extracted_records_2 = {}  # {soc: {...}} 2# 样机记录
        self.extracted_records = self.extracted_records_1  # 兼容引用
        self.project = None
        self._selected_project_path = None
        self.dirty_units = set()
        self._closing = False
        self._close_timer = QTimer(self)
        self._close_timer.setInterval(100)
        self._close_timer.timeout.connect(self._finish_close)
        self._last_unit_modes = {1: 'discharge', 2: 'discharge'}

        self.ocr_tester = None
        self.worker = None
        self.lcd_settings = QSettings('PowerbankTestTool', 'LCDIgnoreRegions')
        self.lcd_ignore_regions = {}
        self.lcd_settings_errors = set()
        for unit in (1, 2):
            try:
                self.lcd_ignore_regions[unit] = validate_ignore_regions(json.loads(
                    self.lcd_settings.value(f'unit_{unit}', '[]')))
            except (ValueError, TypeError):
                self.lcd_ignore_regions[unit] = ()
                self.lcd_settings_errors.add(unit)
        self.segment_settings = QSettings('PowerbankTestTool', 'SegmentMode')
        self.segment_paths = {}
        for unit in (1, 2):
            saved = self.segment_settings.value(f'libraries_{unit}')
            legacy = self.segment_settings.value(f'library_{unit}', str(data_home() / f'unit_{unit}'))
            self.segment_paths[unit] = library_paths(json.loads(saved) if saved is not None else legacy)

        self.init_ui()
        self._wheel_guard = disable_wheel_changes(self)
        self.load_excel_groups()
        self.reload_projects()
        self.resource_monitor = ResourceMonitor(self)
        self.resource_monitor.updated.connect(self.update_resource_display)
        self.resource_monitor.start()

    def get_ocr_tester(self):
        """延迟加载 OCR 引擎，确保主界面秒开"""
        if self.ocr_tester is None:
            self.ocr_tester = OCREngine()
        return self.ocr_tester

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # 提前初始化视频画布组件
        self.canvas = VideoCanvas()
        self.canvas.roi_changed_signal.connect(self.on_roi_changed)
        self.canvas.roi_rotation_changed_signal.connect(self.on_canvas_rotation_changed)

        # ==========================================
        # 左侧控制面板 (固定宽度 380px)
        # ==========================================
        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # --- 模块1: 视频与测试项设置 ---
        grp_video = QGroupBox("1. 视频与测试项目")
        grp_video_layout = QVBoxLayout(grp_video)

        self.btn_select_video = QPushButton("📂 打开录屏视频文件 (.mp4)")
        self.btn_select_video.setStyleSheet("height: 36px; font-weight: bold; font-size: 13px; background-color: #0078D7; color: white;")
        self.btn_select_video.clicked.connect(self.select_video)
        grp_video_layout.addWidget(self.btn_select_video)

        self.lbl_video_name = QLabel("未选择视频")
        self.lbl_video_name.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_video_name.setWordWrap(True)
        grp_video_layout.addWidget(self.lbl_video_name)

        self.unit_enabled, self.unit_modes, self.unit_groups = {}, {}, {}
        for unit in (1, 2):
            row = QHBoxLayout()
            enabled = QCheckBox(f'{unit}#')
            enabled.setChecked(True)
            mode = QComboBox()
            mode.addItem('放电', 'discharge')
            mode.addItem('充电', 'charge')
            group = QComboBox()
            group.setMinimumWidth(150)
            mode.currentIndexChanged.connect(lambda _, u=unit: self.on_unit_mode_changed(u))
            enabled.toggled.connect(lambda checked, u=unit: self.on_unit_enabled_changed(u, checked))
            row.addWidget(enabled)
            row.addWidget(mode)
            row.addWidget(group, 1)
            grp_video_layout.addLayout(row)
            self.unit_enabled[unit], self.unit_modes[unit], self.unit_groups[unit] = enabled, mode, group
        grp_video_layout.addWidget(QLabel('每台分别选充/放电和对应 Excel 测试项'))

        project_row = QHBoxLayout()
        self.cb_project = QComboBox()
        self.cb_project.currentIndexChanged.connect(self.select_project)
        self.btn_new_project = QPushButton('新建存档')
        self.btn_new_project.clicked.connect(self.new_project)
        self.btn_delete_project = QPushButton('删除存档')
        self.btn_delete_project.clicked.connect(self.delete_current_project)
        project_row.addWidget(QLabel('存档:'))
        project_row.addWidget(self.cb_project, 1)
        project_row.addWidget(self.btn_new_project)
        project_row.addWidget(self.btn_delete_project)
        grp_video_layout.addLayout(project_row)
        self.lbl_project_status = QLabel('未选择存档')
        self.lbl_project_status.setWordWrap(True)
        grp_video_layout.addWidget(self.lbl_project_status)

        left_layout.addWidget(grp_video)

        # --- 模块2: 选区画框设置 (支持 1# 和 2# 双套共 6 个框，支持数显旋转) ---
        grp_roi = QGroupBox("2. 选区画框与朝向设置 (鼠标在右图拖拽绘制)")
        grp_roi_layout = QVBoxLayout(grp_roi)

        # 1# 样机控制区
        lbl_unit1 = QLabel("🟦 1# 样机/设备选区:")
        lbl_unit1.setStyleSheet("font-weight: bold; color: #0078D7; margin-top: 2px;")
        grp_roi_layout.addWidget(lbl_unit1)

        row_soc1 = QHBoxLayout()
        self.rb_roi_soc_1 = QRadioButton("● 1# 电量 (蓝框)")
        self.rb_roi_soc_1.setStyleSheet("color: #0096FF; font-weight: bold;")
        self.cb_rot_soc_1 = QComboBox()
        self.cb_rot_soc_1.setStyleSheet("height: 22px; font-size: 11px;")
        self.cb_rot_soc_1.addItem("正向 0°", 0)
        self.cb_rot_soc_1.addItem("顺时针 90° (向右侧放)", 90)
        self.cb_rot_soc_1.addItem("倒立 180°", 180)
        self.cb_rot_soc_1.addItem("逆时针 270° (向左侧放)", 270)
        self.cb_rot_soc_1.currentIndexChanged.connect(lambda idx: self.on_rot_combo_changed('soc_1', self.cb_rot_soc_1))
        row_soc1.addWidget(self.rb_roi_soc_1, stretch=5)
        row_soc1.addWidget(self.cb_rot_soc_1, stretch=5)
        grp_roi_layout.addLayout(row_soc1)

        self.rb_roi_time_1 = QRadioButton("● 1# 时间 (绿框)")
        self.rb_roi_cap_1 = QRadioButton("● 1# 容量 (橙框)")
        self.rb_roi_time_1.setStyleSheet("color: #00C864; font-weight: bold;")
        self.rb_roi_cap_1.setStyleSheet("color: #FF8C00; font-weight: bold;")
        grp_roi_layout.addWidget(self.rb_roi_time_1)
        grp_roi_layout.addWidget(self.rb_roi_cap_1)

        # 2# 样机控制区
        lbl_unit2 = QLabel("🟪 2# 样机/设备选区 (第二套 3 个框):")
        lbl_unit2.setStyleSheet("font-weight: bold; color: #8A2BE2; margin-top: 6px;")
        grp_roi_layout.addWidget(lbl_unit2)

        row_soc2 = QHBoxLayout()
        self.rb_roi_soc_2 = QRadioButton("● 2# 电量 (紫框)")
        self.rb_roi_soc_2.setStyleSheet("color: #AA55FF; font-weight: bold;")
        self.cb_rot_soc_2 = QComboBox()
        self.cb_rot_soc_2.setStyleSheet("height: 22px; font-size: 11px;")
        self.cb_rot_soc_2.addItem("正向 0°", 0)
        self.cb_rot_soc_2.addItem("顺时针 90° (向右侧放)", 90)
        self.cb_rot_soc_2.addItem("倒立 180°", 180)
        self.cb_rot_soc_2.addItem("逆时针 270° (向左侧放)", 270)
        self.cb_rot_soc_2.currentIndexChanged.connect(lambda idx: self.on_rot_combo_changed('soc_2', self.cb_rot_soc_2))
        row_soc2.addWidget(self.rb_roi_soc_2, stretch=5)
        row_soc2.addWidget(self.cb_rot_soc_2, stretch=5)
        grp_roi_layout.addLayout(row_soc2)

        self.rb_roi_time_2 = QRadioButton("● 2# 时间 (青框)")
        self.rb_roi_cap_2 = QRadioButton("● 2# 容量 (红框)")
        self.rb_roi_time_2.setStyleSheet("color: #00D2D2; font-weight: bold;")
        self.rb_roi_cap_2.setStyleSheet("color: #FF4B4B; font-weight: bold;")
        grp_roi_layout.addWidget(self.rb_roi_time_2)
        grp_roi_layout.addWidget(self.rb_roi_cap_2)

        # 单选按钮互斥组
        self.roi_btn_group = QButtonGroup()
        for rb, r_type in [
            (self.rb_roi_soc_1, 'soc_1'), (self.rb_roi_time_1, 'time_1'), (self.rb_roi_cap_1, 'cap_1'),
            (self.rb_roi_soc_2, 'soc_2'), (self.rb_roi_time_2, 'time_2'), (self.rb_roi_cap_2, 'cap_2')
        ]:
            self.roi_btn_group.addButton(rb)
            rb.toggled.connect(lambda checked, t=r_type: self.on_roi_rb_toggled(checked, t))

        self.rb_roi_soc_1.setChecked(True)
        self.canvas.set_active_roi_type('soc_1')

        # 旋转与清除工具按钮行
        btn_tools_layout = QHBoxLayout()
        self.btn_rotate_roi = QPushButton("🔄 选区旋转90°")
        self.btn_rotate_roi.setStyleSheet("height: 25px; font-size: 11px; font-weight: bold; background-color: #e3f2fd; color: #0d47a1; border: 1px solid #90caf9;")
        self.btn_rotate_roi.clicked.connect(self.rotate_active_roi)
        self.btn_clear_unit1 = QPushButton("清空1#框")
        self.btn_clear_unit1.setStyleSheet("height: 25px; font-size: 11px;")
        self.btn_clear_unit1.clicked.connect(lambda: self.clear_unit_rois(1))
        self.btn_clear_unit2 = QPushButton("清空2#框")
        self.btn_clear_unit2.setStyleSheet("height: 25px; font-size: 11px;")
        self.btn_clear_unit2.clicked.connect(lambda: self.clear_unit_rois(2))

        btn_tools_layout.addWidget(self.btn_rotate_roi)
        btn_tools_layout.addWidget(self.btn_clear_unit1)
        btn_tools_layout.addWidget(self.btn_clear_unit2)
        grp_roi_layout.addLayout(btn_tools_layout)
        rotation_tip = QLabel('拖动框上圆形手柄可360°旋转；双击手柄恢复0°。')
        rotation_tip.setWordWrap(True)
        grp_roi_layout.addWidget(rotation_tip)
        # 选帧滑块
        grp_roi_layout.addWidget(QLabel("选帧预览 (滑动选择清晰画面):"))
        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setEnabled(False)
        self.slider_frame.valueChanged.connect(self.on_slider_frame_changed)
        grp_roi_layout.addWidget(self.slider_frame)

        self.lbl_frame_info = QLabel("当前帧: 0 / 0 (00:00:00)")
        self.lbl_frame_info.setStyleSheet("color: #666; font-size: 11px;")
        grp_roi_layout.addWidget(self.lbl_frame_info)

        seek_row = QHBoxLayout()
        self.btn_seek_back = QPushButton("◀ 后退")
        self.sp_seek_seconds = QDoubleSpinBox()
        self.sp_seek_seconds.setRange(0.1, 3600.0)
        self.sp_seek_seconds.setDecimals(1)
        self.sp_seek_seconds.setValue(1.0)
        self.sp_seek_seconds.setSuffix(" 秒")
        self.sp_seek_seconds.setToolTip("每次前进或后退的秒数；不改变分析抽帧间隔")
        self.btn_seek_forward = QPushButton("前进 ▶")
        for button in (self.btn_seek_back, self.btn_seek_forward):
            button.setEnabled(False)
        self.btn_seek_back.clicked.connect(lambda: self.step_preview(-1))
        self.btn_seek_forward.clicked.connect(lambda: self.step_preview(1))
        seek_row.addWidget(self.btn_seek_back)
        seek_row.addWidget(self.sp_seek_seconds)
        seek_row.addWidget(self.btn_seek_forward)
        grp_roi_layout.addLayout(seek_row)

        self.btn_locate_record = QPushButton("定位所选记录的原视频")
        self.btn_locate_record.setEnabled(False)
        self.btn_locate_record.clicked.connect(self.locate_selected_record)
        grp_roi_layout.addWidget(self.btn_locate_record)

        self.btn_saved_records = QPushButton("查看所选测试项的存档记录")
        self.btn_saved_records.clicked.connect(self.load_saved_records)
        grp_roi_layout.addWidget(self.btn_saved_records)

        # 测试 OCR 按钮
        self.btn_test_ocr = QPushButton("🔍 测试当前帧框选识别 (1# / 2#)")
        self.btn_test_ocr.setStyleSheet("height: 32px; background-color: #555; color: white;")
        self.btn_test_ocr.clicked.connect(self.test_current_frame_ocr)
        grp_roi_layout.addWidget(self.btn_test_ocr)

        left_layout.addWidget(grp_roi)

        # --- 模块3: 参数设置 ---
        grp_params = QGroupBox("3. 提取参数设置")
        grp_params_layout = QGridLayout(grp_params)

        grp_params_layout.addWidget(QLabel("抽帧间隔(秒):"), 0, 0)
        self.sp_interval = QDoubleSpinBox()
        self.sp_interval.setRange(0.2, 10.0)
        self.sp_interval.setValue(1.0)
        self.sp_interval.setSingleStep(0.5)
        grp_params_layout.addWidget(self.sp_interval, 0, 1)

        grp_params_layout.addWidget(QLabel("防抖确认帧数:"), 1, 0)
        self.sp_debounce = QSpinBox()
        self.sp_debounce.setRange(1, 5)
        self.sp_debounce.setValue(2)
        grp_params_layout.addWidget(self.sp_debounce, 1, 1)

        grp_params_layout.addWidget(QLabel("容量数值单位:"), 2, 0)
        self.cb_cap_unit = QComboBox()
        self.cb_cap_unit.addItems(["Ah (上位机显示为 Ah)", "mAh (上位机为 mAh, 自动除以1000)"])
        grp_params_layout.addWidget(self.cb_cap_unit, 2, 1)

        grp_params_layout.addWidget(QLabel("电量屏字形:"), 3, 0)
        self.cb_digit_type = QComboBox()
        self.cb_digit_type.addItem('LCD / 普通数字', 1)
        self.cb_digit_type.addItem('数码管样本库', SEGMENT_MODE)
        grp_params_layout.addWidget(self.cb_digit_type, 3, 1)
        self.btn_segment_dataset = QPushButton('数码管视频采集 / 标注 / 选择样本库')
        self.btn_segment_dataset.setEnabled(False)
        self.btn_segment_dataset.setToolTip('仅在“数码管专用”模式开放；普通 LED/OCR 不使用样本库。')
        self.btn_segment_dataset.clicked.connect(self.open_segment_dataset)
        grp_params_layout.addWidget(self.btn_segment_dataset, 4, 0, 1, 2)
        self.lbl_segment_templates = QLabel()
        self.lbl_segment_templates.setWordWrap(True)
        self.lbl_segment_templates.setVisible(False)
        grp_params_layout.addWidget(self.lbl_segment_templates, 5, 0, 1, 2)
        self.lcd_ignore_buttons = {}
        for unit in (1, 2):
            button = QPushButton()
            button.clicked.connect(lambda _, u=unit: self.edit_lcd_ignore_regions(u))
            self.lcd_ignore_buttons[unit] = button
            grp_params_layout.addWidget(button, 6, unit-1)
        self.lbl_lcd_ignore = QLabel('LCD 电量忽略图标：设置后请保持框选范围一致；换屏幕或重新画框后需检查。')
        self.lbl_lcd_ignore.setWordWrap(True)
        grp_params_layout.addWidget(self.lbl_lcd_ignore, 7, 0, 1, 2)
        self.cb_digit_type.currentIndexChanged.connect(self.update_segment_controls)

        left_layout.addWidget(grp_params)

        grp_resource = QGroupBox('资源占用（整机）')
        resource_layout = QVBoxLayout(grp_resource)
        self.lbl_resource = QLabel('CPU / GPU：正在读取…')
        resource_layout.addWidget(self.lbl_resource)
        self.cb_cpu_high_speed = QCheckBox('CPU 高速模式（预留部分算力）')
        performance_settings = QSettings('PowerbankTestTool', 'Performance')
        self.cb_cpu_high_speed.setChecked(performance_settings.value('high_speed', False, type=bool))
        self.cb_cpu_high_speed.setToolTip(
            '自动使用约85%的逻辑线程，加速数码管样本匹配和OCR。\n'
            '实际CPU占用取决于视频读取和识别任务，不保证固定百分比。下次开始分析时生效。')
        self.cb_cpu_high_speed.toggled.connect(self.on_high_speed_changed)
        resource_layout.addWidget(self.cb_cpu_high_speed)
        self.lbl_cpu_budget = QLabel(f'高速模式使用 {high_speed_threads()} / {os.cpu_count() or 1} 个逻辑线程')
        resource_layout.addWidget(self.lbl_cpu_budget)
        cpu_row = QHBoxLayout()
        cpu_row.addWidget(QLabel('识别计算线程数:'))
        self.cb_cpu_threads = QComboBox()
        self.cb_cpu_threads.addItem('自动', None)
        for count in range(1, (os.cpu_count() or 1) + 1):
            self.cb_cpu_threads.addItem(str(count), count)
        saved_threads = QSettings('PowerbankTestTool', 'Performance').value('cpu_threads')
        if saved_threads not in (None, ''):
            try:
                selected = self.cb_cpu_threads.findData(int(saved_threads))
                if selected >= 0:
                    self.cb_cpu_threads.setCurrentIndex(selected)
            except ValueError:
                pass
        self.cb_cpu_threads.currentIndexChanged.connect(lambda: QSettings(
            'PowerbankTestTool', 'Performance').setValue(
                'cpu_threads', self.cb_cpu_threads.currentData() or ''))
        cpu_row.addWidget(self.cb_cpu_threads, 1)
        resource_layout.addLayout(cpu_row)
        self.update_segment_controls()
        left_layout.addWidget(grp_resource)

        # --- 模块4: 运行与导出 ---
        grp_action = QGroupBox("4. 执行与导出")
        grp_action_layout = QVBoxLayout(grp_action)

        self.btn_start = QPushButton("▶ 开始自动快进分析")
        self.btn_start.setStyleSheet("height: 42px; font-size: 14px; font-weight: bold; background-color: #28a745; color: white;")
        self.btn_start.clicked.connect(self.start_analysis)
        grp_action_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹ 停止分析")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("height: 30px; background-color: #dc3545; color: white;")
        self.btn_stop.clicked.connect(self.stop_analysis)
        grp_action_layout.addWidget(self.btn_stop)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        grp_action_layout.addWidget(line)

        self.btn_save_project = QPushButton('💾 保存本次结果到存档')
        self.btn_save_project.clicked.connect(self.save_to_project)
        grp_action_layout.addWidget(self.btn_save_project)
        self.btn_open_report = QPushButton('📂 打开存档 Excel（检查 / 插图）')
        self.btn_open_report.clicked.connect(self.open_project_report)
        grp_action_layout.addWidget(self.btn_open_report)
        self.btn_plan = QPushButton('☑ 选择计划做的测试项')
        self.btn_plan.clicked.connect(self.edit_project_plan)
        grp_action_layout.addWidget(self.btn_plan)

        self.btn_export_excel = QPushButton("📊 导出存档完整 Excel 报告")
        self.btn_export_excel.setStyleSheet("height: 38px; font-weight: bold; background-color: #0078D7; color: white;")
        self.btn_export_excel.clicked.connect(self.export_to_excel)
        grp_action_layout.addWidget(self.btn_export_excel)

        self.btn_export_csv = QPushButton("📑 导出完整数据为 CSV")
        self.btn_export_csv.setStyleSheet("height: 32px; background-color: #6c757d; color: white;")
        self.btn_export_csv.clicked.connect(self.export_to_csv)
        grp_action_layout.addWidget(self.btn_export_csv)

        left_layout.addWidget(grp_action)
        left_layout.addStretch()

        # ==========================================
        # 右侧内容区 (上方画布，下方进度与两个数据表)
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # 视频画布组件
        right_layout.addWidget(self.canvas, stretch=5)

        # 进度与状态行
        status_box = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("height: 20px; text-align: center;")
        status_box.addWidget(self.progress_bar, stretch=4)

        self.lbl_status = QLabel("状态: 就绪")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #333; padding: 0 10px;")
        status_box.addWidget(self.lbl_status, stretch=4)
        self.lbl_resource_live = QLabel('CPU / GPU：读取中')
        self.lbl_resource_live.setStyleSheet('font-weight: bold; color: #16704a;')
        status_box.addWidget(self.lbl_resource_live, stretch=2)
        right_layout.addLayout(status_box)

        # 下方 28 个目标点实时数据表格区 (双表并排)
        title_bar = QHBoxLayout()
        table_label = QLabel("📋 28 个关键节点测试数据捕获表 (点击时刻跳转预览，双击可直接修改)")
        table_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #1a5fb4;")
        title_bar.addWidget(table_label)
        title_bar.addStretch()

        self.view_combo = QComboBox()
        self.view_combo.addItems(["并排显示 1# 和 2# 数据表", "仅显示 1# 表格", "仅显示 2# 表格"])
        self.view_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        title_bar.addWidget(QLabel("视图切换:"))
        title_bar.addWidget(self.view_combo)
        right_layout.addLayout(title_bar)

        # 两个表格放在 QSplitter 中实现并排布局
        self.splitter_tables = QSplitter(Qt.Horizontal)

        # 1# 表格容器
        self.container_table_1 = QWidget()
        layout_t1 = QVBoxLayout(self.container_table_1)
        layout_t1.setContentsMargins(0, 0, 0, 0)
        layout_t1.setSpacing(4)
        lbl_t1 = QLabel("🟦 1# 测试机数据表 (蓝绿橙选区)")
        lbl_t1.setStyleSheet("font-weight: bold; color: #0078D7; font-size: 12px;")
        layout_t1.addWidget(lbl_t1)
        self.table_1 = QTableWidget()
        self.setup_table_properties(self.table_1, unit_id=1)
        layout_t1.addWidget(self.table_1)

        # 2# 表格容器
        self.container_table_2 = QWidget()
        layout_t2 = QVBoxLayout(self.container_table_2)
        layout_t2.setContentsMargins(0, 0, 0, 0)
        layout_t2.setSpacing(4)
        lbl_t2 = QLabel("🟪 2# 测试机数据表 (紫青红选区)")
        lbl_t2.setStyleSheet("font-weight: bold; color: #8A2BE2; font-size: 12px;")
        layout_t2.addWidget(lbl_t2)
        self.table_2 = QTableWidget()
        self.setup_table_properties(self.table_2, unit_id=2)
        layout_t2.addWidget(self.table_2)

        # 保持旧属性引用
        self.table = self.table_1

        self.splitter_tables.addWidget(self.container_table_1)
        self.splitter_tables.addWidget(self.container_table_2)
        self.splitter_tables.setStretchFactor(0, 1)
        self.splitter_tables.setStretchFactor(1, 1)

        right_layout.addWidget(self.splitter_tables, stretch=4)

        # 将左侧面板放入滚动区域中，防止在分辨率较低或全屏被右侧表格挤压时内容重叠
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(400)
        scroll_area.setWidget(left_panel)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(right_panel)

        self.refresh_target_tables()

    def on_roi_rb_toggled(self, checked, roi_type):
        if checked:
            self.canvas.set_active_roi_type(roi_type)
            rot = self.canvas.get_roi_rotation(roi_type)
            name_map = {
                'soc_1': '1# 电量', 'time_1': '1# 时间', 'cap_1': '1# 容量',
                'soc_2': '2# 电量', 'time_2': '2# 时间', 'cap_2': '2# 容量'
            }
            rot_str = f" (朝向: {rot}°)" if rot != 0 else ""
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText(f"当前绘制目标: {name_map.get(roi_type, roi_type)}{rot_str}")

    def on_rot_combo_changed(self, roi_type, combo):
        angle = combo.currentData()
        self.canvas.set_roi_rotation(roi_type, angle)
        name = "1# 电量" if roi_type == 'soc_1' else "2# 电量"
        self.lbl_status.setText(f"已设置 {name} 旋转朝向为: {angle}° (若侧放将自动回正)")

    def on_canvas_rotation_changed(self, roi_type, angle):
        if roi_type == 'soc_1':
            idx = self.cb_rot_soc_1.findData(angle)
            if idx >= 0 and self.cb_rot_soc_1.currentIndex() != idx:
                self.cb_rot_soc_1.blockSignals(True)
                self.cb_rot_soc_1.setCurrentIndex(idx)
                self.cb_rot_soc_1.blockSignals(False)
        elif roi_type == 'soc_2':
            idx = self.cb_rot_soc_2.findData(angle)
            if idx >= 0 and self.cb_rot_soc_2.currentIndex() != idx:
                self.cb_rot_soc_2.blockSignals(True)
                self.cb_rot_soc_2.setCurrentIndex(idx)
                self.cb_rot_soc_2.blockSignals(False)

    def rotate_active_roi(self):
        roi_type, new_ang = self.canvas.rotate_active_roi(90)
        if roi_type:
            name_map = {
                'soc_1': '1# 电量', 'time_1': '1# 时间', 'cap_1': '1# 容量',
                'soc_2': '2# 电量', 'time_2': '2# 时间', 'cap_2': '2# 容量'
            }
            name = name_map.get(roi_type, roi_type)
            self.lbl_status.setText(f"🔄 已顺时针旋转【{name}】选区朝向至: {new_ang}°")

    def clear_unit_rois(self, unit):
        self.canvas.clear_unit_rois(unit)
        self.lbl_status.setText(f"已清空 {unit}# 样机的所有选区框")

    def setup_table_properties(self, table, unit_id):
        table.setColumnCount(5)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.cellClicked.connect(lambda r, c, u=unit_id: self.on_table_cell_clicked(r, c, u))
        table.cellChanged.connect(lambda r, c, u=unit_id: self.on_table_cell_changed(r, c, u))

    def on_view_mode_changed(self, index):
        if index == 0:  # 并排显示
            self.container_table_1.show()
            self.container_table_2.show()
            total_w = self.splitter_tables.width()
            self.splitter_tables.setSizes([total_w // 2, total_w // 2])
        elif index == 1:  # 仅看 1#
            self.container_table_1.show()
            self.container_table_2.hide()
        elif index == 2:  # 仅看 2#
            self.container_table_1.hide()
            self.container_table_2.show()

    def load_excel_groups(self):
        """扫描当前目录下的 Excel 模板，读取可用的测试项"""
        path = str(self.project.report_path) if self.project else self.excel_template_path
        if not os.path.exists(path):
            self.lbl_status.setText("提示: 未在当前目录找到默认 Excel 报告模板")
            return

        try:
            self.excel_groups = scan_excel_test_groups(path)
            for unit in (1, 2):
                self.update_group_combobox(unit)
        except Exception as e:
            QMessageBox.warning(self, "模板加载警告", f"读取 Excel 模板失败: {str(e)}")

    def on_unit_mode_changed(self, unit):
        new_mode = self.unit_modes[unit].currentData()
        previous = self._last_unit_modes[unit]
        if new_mode != previous and unit in self.dirty_units:
            if QMessageBox.question(self, '尚未保存',
                    f'{unit}# 当前表格有未存入存档的结果，切换充放电会清空它。继续吗？') != QMessageBox.Yes:
                self.unit_modes[unit].blockSignals(True)
                self.unit_modes[unit].setCurrentIndex(self.unit_modes[unit].findData(previous))
                self.unit_modes[unit].blockSignals(False)
                return
            self.dirty_units.discard(unit)
        self._last_unit_modes[unit] = new_mode
        if unit == 1:
            self.extracted_records_1 = {}
        else:
            self.extracted_records_2 = {}
        self.update_group_combobox(unit)
        self.refresh_target_table(unit)

    def on_unit_enabled_changed(self, unit, checked):
        self.unit_modes[unit].setEnabled(checked)
        self.unit_groups[unit].setEnabled(checked)

    def update_group_combobox(self, unit):
        widget = self.unit_groups[unit]
        old = widget.currentData()
        old_key = group_key(old) if old else None
        widget.blockSignals(True)
        widget.clear()
        mode = self.unit_modes[unit].currentData()
        sheet_key = '放电时间及曲线' if mode == 'discharge' else '充电时间及曲线'
        for group in self.excel_groups.get(sheet_key, []):
            if group['col_offset'] == (0 if unit == 1 else 14):
                widget.addItem(group['name'], group)
        for index in range(widget.count()):
            if group_key(widget.itemData(index)) == old_key:
                widget.setCurrentIndex(index)
                break
        widget.blockSignals(False)

    def reload_projects(self, select_path=None, confirmed=False):
        target = select_path or QSettings('PowerbankTestTool', 'Projects').value('last_project', '')
        self.cb_project.blockSignals(True)
        self.cb_project.clear()
        self.cb_project.addItem('未选择存档', None)
        for item in project_catalog():
            self.cb_project.addItem(item['name'], item['path'])
        index = self.cb_project.findData(str(target)) if target else 0
        self.cb_project.setCurrentIndex(max(0, index))
        self.cb_project.blockSignals(False)
        self.select_project(confirmed=confirmed)

    def _restore_project_choice(self):
        index = self.cb_project.findData(self._selected_project_path)
        self.cb_project.blockSignals(True)
        self.cb_project.setCurrentIndex(max(0, index))
        self.cb_project.blockSignals(False)

    def _can_change_project(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, '分析进行中', '请先停止或完成视频分析，再切换存档。')
            return False
        if self.dirty_units and QMessageBox.question(
                self, '结果尚未保存',
                '当前表格有未存入存档的结果。切换存档会清空这些结果，继续吗？') != QMessageBox.Yes:
            return False
        return True

    def select_project(self, *_, confirmed=False):
        path = self.cb_project.currentData()
        if path == self._selected_project_path:
            self.refresh_project_status()
            return
        if not confirmed and not self._can_change_project():
            self._restore_project_choice()
            return
        try:
            project = ProjectArchive(path) if path else None
            self.project = project
            self._selected_project_path = path
            if path:
                QSettings('PowerbankTestTool', 'Projects').setValue('last_project', path)
            self.extracted_records_1 = {}
            self.extracted_records_2 = {}
            self.extracted_records = self.extracted_records_1
            self.dirty_units.clear()
            self.refresh_target_tables()
            self.load_excel_groups()
            self.refresh_project_status()
        except Exception as exc:
            self._restore_project_choice()
            QMessageBox.warning(self, '打开存档失败', str(exc))

    def new_project(self):
        if not self._can_change_project():
            return
        name, ok = QInputDialog.getText(self, '新建测试存档', '存档名称（例如 PB519）：')
        if not ok or not name.strip():
            return
        template, _ = QFileDialog.getOpenFileName(self, '选择这份存档的 Excel 原始模板',
            self.excel_template_path, 'Excel 文件 (*.xlsx)')
        if not template:
            return
        try:
            project = create_project(name, template)
            self.reload_projects(str(project.root), confirmed=True)
        except Exception as exc:
            QMessageBox.warning(self, '新建存档失败', str(exc))

    def delete_current_project(self):
        if not self.project:
            QMessageBox.information(self, '未选择存档', '请先选择要删除的存档。')
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, '分析进行中', '请先停止或完成视频分析，再删除存档。')
            return
        name = self.project.metadata.get('name', self.project.root.name)
        count = len(self.project.metadata.get('entries', {}))
        warning = '\n当前表格还有未存入存档的结果。' if self.dirty_units else ''
        message = (f'确定删除存档“{name}”吗？\n\n'
                   f'已保存测试项：{count} 项\n'
                   f'工作目录：{self.project.root}{warning}\n\n'
                   '存档内的工作版 Excel 和记录将被删除，此操作不能撤销。')
        if QMessageBox.question(self, '确认删除存档', message,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            delete_project(self.project.root)
            QSettings('PowerbankTestTool', 'Projects').remove('last_project')
            self.project = None
            self._selected_project_path = None
            self.extracted_records_1 = {}
            self.extracted_records_2 = {}
            self.extracted_records = self.extracted_records_1
            self.dirty_units.clear()
            self.refresh_target_tables()
            self.reload_projects(confirmed=True)
            QMessageBox.information(self, '删除完成', f'存档“{name}”已删除。')
        except Exception as exc:
            QMessageBox.warning(self, '删除失败', str(exc))

    def refresh_project_status(self):
        active = bool(self.project)
        self.btn_delete_project.setEnabled(active)
        for button in (self.btn_save_project, self.btn_open_report, self.btn_plan, self.btn_export_excel):
            button.setEnabled(active)
        if not active:
            self.lbl_project_status.setText('未选择存档；请新建或选择已有存档。')
            return
        entries = self.project.metadata['entries']
        planned = set(self.project.metadata.get('planned', []))
        pending = len(planned - set(entries))
        self.lbl_project_status.setText(
            f'{self.project.metadata["name"]}：已保存 {len(entries)} 项'
            + (f'，计划中待完成 {pending} 项' if planned else '')
            + f'\n工作表：{self.project.report_path}')

    def edit_project_plan(self):
        if not self.project:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('选择计划做的测试项')
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('勾选计划做的充电和放电项目。已保存结果不会因取消计划而删除。'))
        listing = QListWidget()
        planned = set(self.project.metadata.get('planned', []))
        completed = set(self.project.metadata['entries'])
        for groups in self.project.groups.values():
            for group in groups:
                key = group_key(group)
                item = QListWidgetItem(group['name'] + ('  ✓ 已保存' if key in completed else '  · 未完成'))
                item.setData(Qt.UserRole, key)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if key in planned else Qt.Unchecked)
                listing.addItem(item)
        layout.addWidget(listing)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(500, 550)
        if dialog.exec_() == QDialog.Accepted:
            self.project.set_planned([listing.item(i).data(Qt.UserRole)
                for i in range(listing.count()) if listing.item(i).checkState() == Qt.Checked])
            self.refresh_project_status()

    def save_to_project(self):
        if not self.project:
            QMessageBox.warning(self, '未选择存档', '请先新建或选择测试存档。')
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, '正在分析', '请等视频分析完成，再核对并保存结果。')
            return
        assignments = []
        for unit in (1, 2):
            if not self.unit_enabled[unit].isChecked():
                continue
            records = self.get_table_data_for_export(unit)
            if not records:
                continue
            group = self.unit_groups[unit].currentData()
            if not group:
                QMessageBox.warning(self, '缺少测试项', f'{unit}# 尚未选择 Excel 测试项。')
                return
            assignments.append({'unit': unit, 'mode': self.unit_modes[unit].currentData(),
                'group': group, 'records': records})
        if not assignments:
            QMessageBox.warning(self, '没有结果', '参与测试的样机尚无可保存数据。')
            return
        repeated = [item['group']['name'] for item in assignments
            if group_key(item['group']) in self.project.metadata['entries']]
        if repeated and QMessageBox.question(self, '替换已有结果',
                '以下测试项已有结果，确定用当前表格替换吗？\n' + '\n'.join(repeated)) != QMessageBox.Yes:
            return
        try:
            counts = self.project.save_results(assignments, self.video_path)
            self.dirty_units.difference_update(counts)
            self.refresh_project_status()
            QMessageBox.information(self, '已存入存档', '已保存：' + '，'.join(f'{u}# {n} 点' for u, n in counts.items()))
        except Exception as exc:
            QMessageBox.warning(self, '保存失败', f'{exc}\n如果 Excel 正在打开，请先在 Excel 中保存并关闭，再重试。')

    def open_project_report(self):
        if self.project and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.report_path))):
            QMessageBox.warning(self, '打开失败', str(self.project.report_path))

    def update_resource_display(self, info):
        cpu = f"{info['cpu_percent']}%" if info['cpu_percent'] is not None else '不可用'
        gpu = (f"{info['gpu_name']} {info['gpu_percent']}%" if info['gpu_name']
               else 'GPU 占用不可用')
        physical = info['physical_cores'] or '未知'
        self.lbl_resource.setText(f'CPU {cpu} · {physical} 核 / {info["logical_threads"]} 线程\n{gpu}')
        self.lbl_resource_live.setText(f'CPU {cpu} · GPU {info["gpu_percent"]}%' if info['gpu_percent'] is not None
                                       else f'CPU {cpu} · GPU 不可用')

    def load_saved_records(self):
        if not self.project or not self._can_change_project():
            return
        self.dirty_units.clear()
        self.refresh_target_tables(restore=True)
        self.lbl_status.setText('已读取所选 Excel 测试项的存档记录，点击行可查看识别截图。')

    def refresh_target_tables(self, restore=False):
        for unit in (1, 2):
            self.refresh_target_table(unit, restore=restore)

    def refresh_target_table(self, unit, restore=False):
        """Each unit has its own charge/discharge target sequence."""
        mode = self.unit_modes[unit].currentData()
        cap_title = "放电容量 (Ah)" if mode == 'discharge' else "充电容量 (Ah)"
        target_socs = get_target_soc_list(mode)

        for unit_id, table in [(unit, self.table_1 if unit == 1 else self.table_2)]:
            table.blockSignals(True)
            table.setHorizontalHeaderLabels([
                "目标电量",
                "测试时间 (双击修改)",
                f"{cap_title}",
                "视频位置 (点击看记录画面)",
                "状态"
            ])
            table.setRowCount(len(target_socs))

            for row, soc in enumerate(target_socs):
                item_soc = QTableWidgetItem(f"{soc}%")
                item_soc.setTextAlignment(Qt.AlignCenter)
                item_soc.setFont(QFont("Segoe UI", 9, QFont.Bold))
                item_soc.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, 0, item_soc)

                item_time = QTableWidgetItem("--")
                item_time.setTextAlignment(Qt.AlignCenter)
                item_time.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                table.setItem(row, 1, item_time)

                item_cap = QTableWidgetItem("--")
                item_cap.setTextAlignment(Qt.AlignCenter)
                item_cap.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                table.setItem(row, 2, item_cap)

                item_sec = QTableWidgetItem("--")
                item_sec.setTextAlignment(Qt.AlignCenter)
                item_sec.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item_sec.setToolTip(f"点击此行可立即跳转视频画面预览 ({unit_id}#)")
                table.setItem(row, 3, item_sec)

                item_status = QTableWidgetItem("待检测")
                item_status.setTextAlignment(Qt.AlignCenter)
                item_status.setForeground(QColor(150, 150, 150))
                item_status.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, 4, item_status)

            table.blockSignals(False)
            records = self.extracted_records_1 if unit == 1 else self.extracted_records_2
            records.clear()
            group = self.unit_groups[unit].currentData()
            entry = self.project.metadata['entries'].get(group_key(group)) if restore and self.project and group else None
            if entry:
                dirty = unit in self.dirty_units
                for soc, stored in entry.get('records', {}).items():
                    record = self.project.resolve_record(stored, source_video=entry.get('video_path', ''))
                    record.update(soc=int(soc), time_obj=self.parse_time_str(record.get('time_str')), is_target=True)
                    self.on_worker_point_detected(unit, record)
                if not dirty:
                    self.dirty_units.discard(unit)
            self._selected_record_preview = None
            self.btn_locate_record.setEnabled(False)

    def select_video(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择录屏视频", "", "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*.*)"
            )
            if not file_path:
                return

            self.video_path = file_path
            self._preview_frame_valid = False
            self.lbl_video_name.setText(os.path.basename(file_path))

            if self.video_cap:
                self.video_cap.release()

            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开此视频文件，请检查格式或文件权限！")
                return

            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0

            self.slider_frame.setEnabled(True)
            self.slider_frame.setRange(0, max(0, self.total_frames - 1))
            self.slider_frame.setValue(0)

            # 读取第 1 帧
            self.load_frame(0)
            self.lbl_status.setText("视频加载成功，请在画面上框选 1# 和 2# 样机区域。")
        except Exception as e:
            QMessageBox.critical(self, "加载视频异常", f"加载视频失败: {str(e)}\n\n{traceback.format_exc()}")

    def on_slider_frame_changed(self, frame_idx):
        self.load_frame(frame_idx)

    def load_frame(self, frame_idx):
        self._preview_frame_valid = False
        if not self.video_cap:
            return
        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()
            if ret and frame is not None:
                self.current_frame_cv = frame
                self.canvas.set_frame(frame)
                self.canvas.setEnabled(True)
                self.btn_test_ocr.setEnabled(True)
                sec = self.video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                self.set_preview_position(self.video_path, sec)
                m, s = divmod(int(sec), 60)
                h, m = divmod(m, 60)
                self.lbl_frame_info.setText(f"当前帧: {frame_idx} / {self.total_frames} ({h:02d}:{m:02d}:{s:02d})")
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")

    def set_preview_position(self, source, seconds):
        self._preview_frame_valid = True
        self._preview_source = source
        self._preview_seconds = float(seconds)
        enabled = bool(source and os.path.isfile(source))
        self.btn_seek_back.setEnabled(enabled)
        self.btn_seek_forward.setEnabled(enabled)

    def step_preview(self, direction):
        source = getattr(self, '_preview_source', None) or self.video_path
        if not source or not os.path.isfile(source):
            self.lbl_status.setText('请先打开视频，或连接当前记录的原视频。')
            return
        seconds = getattr(self, '_preview_seconds', 0.)
        requested = max(0., seconds + direction * self.sp_seek_seconds.value())
        cap = cv2.VideoCapture(source)
        try:
            if not cap.isOpened():
                raise ValueError('无法打开原视频')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError('视频没有可用帧')
            target = min(requested, max(0., (count - 1) / fps))
            ok, frame = read_frame_at(cap, target)
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
                ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError('无法读取目标画面')
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            self.current_frame_cv = frame
            self.canvas.set_frame(frame)
            self.set_preview_position(source, actual)
            same_video = bool(self.video_path and os.path.normcase(os.path.abspath(source)) ==
                              os.path.normcase(os.path.abspath(self.video_path)))
            self.canvas.setEnabled(same_video)
            self.btn_test_ocr.setEnabled(same_video)
            if same_video:
                self.slider_frame.blockSignals(True)
                self.slider_frame.setValue(index)
                self.slider_frame.blockSignals(False)
            self.lbl_frame_info.setText(f'视频位置 {actual:.3f}s · 第 {index} 帧')
            self.lbl_status.setText(f'已按 {self.sp_seek_seconds.value():g} 秒步长移动预览画面。')
        except (OSError, ValueError, cv2.error) as exc:
            self.lbl_status.setText(f'移动预览失败：{exc}')
        finally:
            cap.release()

    def on_roi_changed(self, roi_type, roi):
        names = {
            'soc_1': '1# 电量(蓝)', 'time_1': '1# 时间(绿)', 'cap_1': '1# 容量(橙)',
            'soc_2': '2# 电量(紫)', 'time_2': '2# 时间(青)', 'cap_2': '2# 容量(红)'
        }
        self.lbl_status.setText(f"已更新选区: {names.get(roi_type, roi_type)} {roi}")

    def validate_selected_rois(self, has_1, has_2):
        if self.current_frame_cv is None:
            return False
        height, width = self.current_frame_cv.shape[:2]
        for unit, enabled in ((1, has_1), (2, has_2)):
            if enabled:
                for kind in ('soc', 'time', 'cap'):
                    key = f'{kind}_{unit}'
                    if not roi_in_bounds(self.canvas.rois[key], width, height):
                        QMessageBox.warning(self, '选区超出边界',
                            f'{self.canvas.names[key]}的旋转框超出视频画面，请先调整红色选区。')
                        return False
        return True

    def on_table_cell_clicked(self, row, col, unit_id=1):
        """点击任意表格中的发生时刻或行时，视频画面与滑块立即跳转至对应时间点"""
        table = self.table_1 if unit_id == 1 else self.table_2
        sec_item = table.item(row, 3)
        if not sec_item:
            return
        text = sec_item.text().strip()
        if not text or text == '--':
            return

        sec_clean = text.lower().replace('s', '').replace('秒', '').strip()
        try:
            video_sec = float(sec_clean)
        except ValueError:
            return

        records = self.extracted_records_1 if unit_id == 1 else self.extracted_records_2
        soc = int(table.item(row, 0).text().replace('%', ''))
        record = dict(records.get(soc, {'video_sec': video_sec}))
        record.setdefault('source_video', self.video_path)
        self._selected_record_preview = record
        self.btn_locate_record.setEnabled(bool(record.get('source_video')))
        snapshot = load_snapshot(record)
        if snapshot is not None:
            self.current_frame_cv = snapshot
            self.canvas.set_frame(snapshot)
            self.canvas.setEnabled(False)
            self.btn_test_ocr.setEnabled(False)
            self.set_preview_position(record.get('source_video'), record['video_sec'])
            self.lbl_frame_info.setText(f"识别时原始截图 · 视频位置 {record['video_sec']:.3f}s")
            self.lbl_status.setText(f"{unit_id}# {soc}%：显示识别当时的画面；测试时间与视频位置分别记录。")
        else:
            self.locate_selected_record()

    def locate_selected_record(self):
        record = getattr(self, '_selected_record_preview', None)
        if not record:
            return
        source = record.get('source_video')
        if not source or not os.path.isfile(source):
            self.lbl_status.setText('原视频不存在；请连接原视频所在磁盘后重试。')
            if load_snapshot(record) is None:
                self.lbl_frame_info.setText('无法预览所选记录，当前画面未更新')
            return
        cap = cv2.VideoCapture(source)
        try:
            if not cap.isOpened():
                raise ValueError('原视频无法打开')
            frame, verified = locate_record(cap, record)
            if frame is None:
                self.lbl_status.setText('未找到与记录截图一致的原视频帧，保留当前画面；视频可能已变化。')
                return
            self.current_frame_cv = frame
            self.canvas.set_frame(frame)
            self.canvas.setEnabled(False)
            self.btn_test_ocr.setEnabled(False)
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            self.set_preview_position(source, actual)
            index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            if self.video_path and os.path.normcase(os.path.abspath(source)) == os.path.normcase(os.path.abspath(self.video_path)):
                self.slider_frame.blockSignals(True)
                self.slider_frame.setValue(index)
                self.slider_frame.blockSignals(False)
            label = '已核对：与记录截图一致' if verified else '旧记录或截图缺失，定位可能有偏差'
            self.lbl_frame_info.setText(f'视频位置 {actual:.3f}s · {label}')
            self.lbl_status.setText(label + '；拖动预览滑块可回到当前加载视频并继续框选。')
        except (OSError, ValueError, cv2.error) as exc:
            self.lbl_status.setText(f'定位失败：{exc}')
        finally:
            cap.release()

    @staticmethod
    def parse_time_str(time_str):
        """容错解析时间字符串为 time 对象"""
        if not time_str or time_str == '--':
            return None
        s = time_str.strip().replace('：', ':').replace(';', ':').replace(' ', '')
        parts = s.split(':')
        try:
            nums = [int(p) for p in parts]
            if len(nums) == 3:
                return time(nums[0], nums[1], nums[2])
            elif len(nums) == 2:
                return time(0, nums[0], nums[1])
            elif len(nums) == 1:
                return time(0, 0, nums[0])
        except Exception:
            return None
        return None

    def on_table_cell_changed(self, row, col, unit_id=1):
        """当用户在表格中手动修改时间或容量时，实时同步缓存并更新状态提示"""
        if col not in (1, 2):
            return

        table = self.table_1 if unit_id == 1 else self.table_2
        records = self.extracted_records_1 if unit_id == 1 else self.extracted_records_2

        soc_item = table.item(row, 0)
        if not soc_item:
            return
        soc_str = soc_item.text().replace('%', '').strip()
        if not soc_str.isdigit():
            return
        soc = int(soc_str)

        time_item = table.item(row, 1)
        cap_item = table.item(row, 2)
        time_text = time_item.text().strip() if time_item else ''
        cap_text = cap_item.text().strip() if cap_item else ''

        if (time_text and time_text != '--') or (cap_text and cap_text != '--'):
            self.dirty_units.add(unit_id)
            time_obj = self.parse_time_str(time_text)
            try:
                cap_val = float(cap_text.replace('Ah', '').replace('mAh', '').strip())
            except ValueError:
                cap_val = 0.0

            rec = records.get(soc, {})
            rec['soc'] = soc
            rec['time_str'] = time_text
            if time_obj:
                rec['time_obj'] = time_obj
            rec['capacity'] = cap_val
            records[soc] = rec

            # 标记为已手动修改
            status_item = table.item(row, 4)
            if status_item:
                table.blockSignals(True)
                status_item.setText("✏️ 已手动修改")
                status_item.setForeground(QColor(0, 102, 204))
                table.blockSignals(False)

    def update_segment_controls(self):
        running = bool(self.worker and self.worker.isRunning())
        self.cb_cpu_high_speed.setEnabled(not running)
        self.cb_cpu_threads.setEnabled(not running and not self.cb_cpu_high_speed.isChecked())
        lcd = self.cb_digit_type.currentData() != SEGMENT_MODE
        self.lbl_lcd_ignore.setVisible(lcd)
        for unit, button in self.lcd_ignore_buttons.items():
            button.setVisible(lcd)
            button.setEnabled(not running)
            count = '需重设' if unit in self.lcd_settings_errors else str(len(self.lcd_ignore_regions[unit]))
            button.setText(f'{unit}# 忽略图标（{count}）')
        self.btn_segment_dataset.setEnabled(self.cb_digit_type.currentData() == SEGMENT_MODE and not running)
        self.lbl_segment_templates.setVisible(self.cb_digit_type.currentData() == SEGMENT_MODE)
        if self.cb_digit_type.currentData() != SEGMENT_MODE:
            return
        names = {entry['path']: entry['name'] for entry in template_catalog(self.segment_paths[1] + self.segment_paths[2])}
        self.lbl_segment_templates.setText('\n'.join(f'{unit}# 模板：' + ('、'.join(names.get(p, p) for p in self.segment_paths[unit]) or '未选择') for unit in (1, 2)))

    def on_high_speed_changed(self, checked):
        QSettings('PowerbankTestTool', 'Performance').setValue('high_speed', checked)
        self.cb_cpu_threads.setEnabled(not checked and not (self.worker and self.worker.isRunning()))

    def edit_lcd_ignore_regions(self, unit):
        if self.cb_digit_type.currentData() == SEGMENT_MODE or (self.worker and self.worker.isRunning()):
            return
        roi = self.canvas.rois.get(f'soc_{unit}')
        if (self.current_frame_cv is None or not self._preview_frame_valid
                or not self.canvas.isEnabled() or not roi):
            QMessageBox.warning(self, 'LCD 忽略图标', f'请先预览当前视频，并画好 {unit}# 电量框。')
            return
        try:
            image = crop_roi(self.current_frame_cv, roi, self.canvas.get_roi_rotation(f'soc_{unit}'), strict=True)
            dialog = IgnoreRegionsDialog(image, self.lcd_ignore_regions[unit], self, lcd_unit=unit)
            if dialog.exec_() == QDialog.Accepted:
                regions = validate_ignore_regions(dialog.canvas.regions)
                self.lcd_settings.setValue(f'unit_{unit}', json.dumps(regions))
                self.lcd_ignore_regions[unit] = regions
                self.lcd_settings_errors.discard(unit)
                self.update_segment_controls()
                self.lbl_status.setText(f'{unit}# LCD 忽略区域已保存，测试当前帧和正式分析都会使用。')
        except (ValueError, cv2.error) as exc:
            QMessageBox.warning(self, 'LCD 忽略图标设置失败', str(exc))

    def open_segment_dataset(self):
        if self.cb_digit_type.currentData() != SEGMENT_MODE or (self.worker and self.worker.isRunning()):
            return
        dialog = SegmentDatasetDialog(self.segment_paths, self.video_path,
            self.canvas.rois, self.canvas.rotations, self)
        dialog.exec_()
        self.segment_paths = dict(dialog.paths)
        if dialog.apply_rois:
            for key, roi in dialog.apply_rois.items():
                if roi:
                    self.canvas.rois[key] = roi
                    self.canvas.set_roi_rotation(key, dialog.apply_rotations[key])
            self.canvas.update()
        for unit, path in self.segment_paths.items():
            self.segment_settings.setValue(f'libraries_{unit}', json.dumps(path, ensure_ascii=False))
        self.update_segment_controls()

    def segment_matchers(self, has_1, has_2):
        matchers = {}
        for unit, enabled in ((1, has_1), (2, has_2)):
            if enabled:
                matcher = SegmentMatcher(self.segment_paths[unit])
                if not matcher.labels:
                    raise ValueError(f'{unit}# 样本库还没有已确认数字。请先打开“数码管视频采集 / 标注”，选择样本库并确认标签。')
                matchers[unit] = matcher
        return matchers

    def test_current_frame_ocr(self):
        """单帧测试框选识别 (支持 1# 和 2# 独立或同时测试)"""
        if self.current_frame_cv is None:
            QMessageBox.warning(self, "提示", "请先加载视频画面！")
            return

        rois = self.canvas.rois
        has_1 = bool(rois['soc_1'] and rois['time_1'] and rois['cap_1'])
        has_2 = bool(rois['soc_2'] and rois['time_2'] and rois['cap_2'])

        if not has_1 and not has_2:
            QMessageBox.warning(self, "提示", "请至少为 1# 或 2# 样机画好完整的 3 个选区框！\n· 1# 选区: 蓝、绿、橙框\n· 2# 选区: 紫、青、红框")
            return

        unit_mode = 'mAh' if 'mAh' in self.cb_cap_unit.currentText() else 'Ah'
        msg_lines = ["【当前帧框选识别测试结果】\n"]

        if not self.validate_selected_rois(has_1, has_2):
            return

        tester = self.get_ocr_tester()
        digit_type = self.cb_digit_type.currentData()
        matchers = {}
        if digit_type == SEGMENT_MODE:
            try:
                matchers = self.segment_matchers(has_1, has_2)
            except Exception as exc:
                QMessageBox.warning(self, '数码管样本库', str(exc))
                return

        if has_1:
            rot_s1 = self.canvas.get_roi_rotation('soc_1')
            rot_t1 = self.canvas.get_roi_rotation('time_1')
            rot_c1 = self.canvas.get_roi_rotation('cap_1')

            c_soc1 = self.crop_helper(self.current_frame_cv, rois['soc_1'], rot_s1)
            c_time1 = self.crop_helper(self.current_frame_cv, rois['time_1'], rot_t1)
            c_cap1 = self.crop_helper(self.current_frame_cv, rois['cap_1'], rot_c1)

            s1, s1_conf = (matchers[1].extract_soc(c_soc1) if digit_type == SEGMENT_MODE
                           else tester.extract_soc(mask_ocr_regions(c_soc1, self.lcd_ignore_regions[1]), digit_type=digit_type))
            t1_str, _, t1_conf = tester.extract_time(c_time1)
            cap1, raw1, _ = tester.extract_capacity(c_cap1, target_unit=unit_mode)

            rot_desc1 = f" (↻{rot_s1}°已回正)" if rot_s1 != 0 else ""
            msg_lines.append("🟦 1# 样机识别:")
            msg_lines.append(f"  · 电量 (蓝框){rot_desc1}: {s1}% (置信度 {s1_conf:.2f})")
            if digit_type == SEGMENT_MODE:
                msg_lines.append('  · 模板判断：' + matchers[1].last_reason)
            msg_lines.append(f"  · 时间 (绿框): {t1_str} (置信度 {t1_conf:.2f})")
            msg_lines.append(f"  · 容量 (橙框): {cap1} Ah (原文: '{raw1}')\n")
        else:
            msg_lines.append("🟦 1# 样机: 选区未画满 3 个框 (跳过)\n")

        if has_2:
            rot_s2 = self.canvas.get_roi_rotation('soc_2')
            rot_t2 = self.canvas.get_roi_rotation('time_2')
            rot_c2 = self.canvas.get_roi_rotation('cap_2')

            c_soc2 = self.crop_helper(self.current_frame_cv, rois['soc_2'], rot_s2)
            c_time2 = self.crop_helper(self.current_frame_cv, rois['time_2'], rot_t2)
            c_cap2 = self.crop_helper(self.current_frame_cv, rois['cap_2'], rot_c2)

            s2, s2_conf = (matchers[2].extract_soc(c_soc2) if digit_type == SEGMENT_MODE
                           else tester.extract_soc(mask_ocr_regions(c_soc2, self.lcd_ignore_regions[2]), digit_type=digit_type))
            t2_str, _, t2_conf = tester.extract_time(c_time2)
            cap2, raw2, _ = tester.extract_capacity(c_cap2, target_unit=unit_mode)

            rot_desc2 = f" (↻{rot_s2}°已回正)" if rot_s2 != 0 else ""
            msg_lines.append("🟪 2# 样机识别:")
            msg_lines.append(f"  · 电量 (紫框){rot_desc2}: {s2}% (置信度 {s2_conf:.2f})")
            if digit_type == SEGMENT_MODE:
                msg_lines.append('  · 模板判断：' + matchers[2].last_reason)
            msg_lines.append(f"  · 时间 (青框): {t2_str} (置信度 {t2_conf:.2f})")
            msg_lines.append(f"  · 容量 (红框): {cap2} Ah (原文: '{raw2}')\n")
        else:
            msg_lines.append("🟪 2# 样机: 选区未画满 3 个框 (跳过)\n")

        msg_lines.append("若识别数值准确，即可点击【开始自动快进分析】！")
        QMessageBox.information(self, "单帧测试结果", "\n".join(msg_lines))

    @staticmethod
    def crop_helper(frame, roi, rotation_angle=0):
        return crop_roi(frame, roi, rotation_angle)

    def start_analysis(self):
        try:
            if self.worker and self.worker.isRunning():
                return
            if self.dirty_units and QMessageBox.question(self, '结果尚未保存',
                    '当前表格有未存入存档的结果。开始新分析会清空它，继续吗？') != QMessageBox.Yes:
                return
            if not self.video_path:
                QMessageBox.warning(self, "提示", "请先选择视频文件！")
                return

            if self.video_cap and not self.canvas.isEnabled():
                self.load_frame(self.slider_frame.value())
            rois = self.canvas.rois
            has_1 = self.unit_enabled[1].isChecked() and bool(rois['soc_1'] and rois['time_1'] and rois['cap_1'])
            has_2 = self.unit_enabled[2].isChecked() and bool(rois['soc_2'] and rois['time_2'] and rois['cap_2'])

            if not has_1 and not has_2:
                QMessageBox.warning(self, "提示", "请至少为 1# 或 2# 样机画好完整的 3 个选区框！\n· 1# 选区: 蓝、绿、橙框\n· 2# 选区: 紫、青、红框")
                return

            if not self.validate_selected_rois(has_1, has_2):
                return

            modes = {unit: self.unit_modes[unit].currentData() for unit in (1, 2)}
            interval = self.sp_interval.value()
            debounce = self.sp_debounce.value()
            unit_mode = 'mAh' if 'mAh' in self.cb_cap_unit.currentText() else 'Ah'

            digit_type = self.cb_digit_type.currentData()
            if digit_type == SEGMENT_MODE:
                self.segment_matchers(has_1, has_2)

            self.refresh_target_tables()
            self.extracted_records_1 = {}
            self.extracted_records_2 = {}
            self.extracted_records = self.extracted_records_1
            self.dirty_units.clear()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_select_video.setEnabled(False)

            analyzing_desc = []
            if has_1: analyzing_desc.append("1# 样机")
            if has_2: analyzing_desc.append("2# 样机")
            self.lbl_status.setText(f"状态: 正在高速分析 {' 和 '.join(analyzing_desc)} 数据...")

            self.worker = VideoAnalysisWorker(
                video_path=self.video_path,
                roi_soc_1=rois['soc_1'],
                roi_time_1=rois['time_1'],
                roi_cap_1=rois['cap_1'],
                roi_soc_2=rois['soc_2'],
                roi_time_2=rois['time_2'],
                roi_cap_2=rois['cap_2'],
                rotations=self.canvas.rotations,
                modes=modes,
                active_units=[unit for unit, active in ((1, has_1), (2, has_2)) if active],
                sample_interval_sec=interval,
                debounce_count=debounce,
                cap_unit=unit_mode,
                digit_type=digit_type,
                segment_paths=dict(self.segment_paths),
                lcd_ignore_regions=dict(self.lcd_ignore_regions),
                cpu_threads=self.cb_cpu_threads.currentData(),
                high_speed=self.cb_cpu_high_speed.isChecked()
            )
            self.worker.progress_signal.connect(self.on_worker_progress)
            self.worker.point_detected_signal.connect(self.on_worker_point_detected)
            self.worker.review_pending_signal.connect(self.on_worker_review_pending)
            self.worker.finished_signal.connect(self.on_worker_finished)
            self.worker.error_signal.connect(self.on_worker_error)
            self.worker.finished.connect(self.update_segment_controls)
            self.worker.finished.connect(lambda: self.cb_digit_type.setEnabled(True))
            self.worker.start()
            self.update_segment_controls()
            self.cb_digit_type.setEnabled(False)
            self.btn_segment_dataset.setEnabled(False)

        except Exception as e:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_select_video.setEnabled(True)
            QMessageBox.critical(self, "启动分析失败", f"启动分析失败: {str(e)}\n\n{traceback.format_exc()}")

    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.lbl_status.setText("状态: 用户手动停止")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_select_video.setEnabled(True)

    def on_worker_progress(self, cur_sec, total_sec, pct, soc1_str, cap1_str, soc2_str, cap2_str):
        if self._closing:
            return
        self.progress_bar.setValue(pct)
        cur_m, cur_s = divmod(int(cur_sec), 60)
        tot_m, tot_s = divmod(int(total_sec), 60)
        self.lbl_status.setText(
            f"进度: {cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d} ({pct}%) | "
            f"1#:[电量 {soc1_str}, 容量 {cap1_str}] | "
            f"2#:[电量 {soc2_str}, 容量 {cap2_str}]"
        )

    def on_worker_review_pending(self, unit_id, entry):
        if self._closing:
            return
        records = self.extracted_records_1 if unit_id == 1 else self.extracted_records_2
        soc = entry['soc']
        targets = get_target_soc_list(self.unit_modes[unit_id].currentData())
        if soc in records or soc not in targets:
            return
        table = self.table_1 if unit_id == 1 else self.table_2
        row = targets.index(soc)
        table.blockSignals(True)
        # Keep time/capacity empty: suspicious matches are never exportable data.
        table.item(row, 3).setText(f"{entry['video_sec']:.1f}s")
        status = table.item(row, 4)
        status.setText('⚠️ 电量跳变待核对')
        status.setToolTip(entry['reason'])
        status.setForeground(QColor(180, 110, 0))
        table.blockSignals(False)

    def on_worker_point_detected(self, unit_id, record):
        soc = record['soc']
        table = self.table_1 if unit_id == 1 else self.table_2
        if unit_id == 1:
            self.extracted_records_1[soc] = record
        else:
            self.extracted_records_2[soc] = record

        mode = self.unit_modes[unit_id].currentData()
        target_socs = get_target_soc_list(mode)
        if soc in target_socs:
            self.dirty_units.add(unit_id)
            row = target_socs.index(soc)
            table.blockSignals(True)
            table.item(row, 1).setText(record['time_str'])
            table.item(row, 2).setText(f"{record['capacity']:.4f}" if record.get("capacity") is not None else "--")
            table.item(row, 3).setText(f"{record['video_sec']:.1f}s" if isinstance(record.get("video_sec"), (int, float)) else "--")
            item_status = table.item(row, 4)
            item_status.setToolTip('')
            if record.get('time_obj') is None:
                item_status.setText("⚠️ 时间待核对")
                item_status.setForeground(QColor(180, 110, 0))
            else:
                item_status.setText("✅ 已捕获" if record.get("locator_version") else "已记录 · 旧定位")
                item_status.setForeground(QColor(0, 150, 0))
            table.blockSignals(False)

    def on_worker_finished(self, all_data_1, all_data_2):
        if self._closing:
            return
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_video.setEnabled(True)
        self.progress_bar.setValue(100)
        targets_1, targets_2 = self.worker.target_socs[1], self.worker.target_socs[2]
        c1 = len(targets_1.intersection(all_data_1))
        c2 = len(targets_2.intersection(all_data_2))
        pending_1 = sum(rec.get('time_obj') is None for soc, rec in all_data_1.items() if soc in targets_1)
        pending_2 = sum(rec.get('time_obj') is None for soc, rec in all_data_2.items() if soc in targets_2)
        diagnostics = ''
        for unit, info in getattr(self.worker, 'segment_diagnostics', {}).items():
            targets = self.worker.target_socs[unit]
            missing = sorted(targets - set((all_data_1 if unit == 1 else all_data_2)))
            absent = sorted(targets - set(info['labels']))
            reasons = '；'.join(f'{reason} {count} 帧' for reason, count in info['rejected'].items()) or '无'
            diagnostics += (f'\n{unit}# 未捕获：{missing}\n尚无有效模板：{absent}\n拒绝原因：{reasons}\n')
            records = all_data_1 if unit == 1 else all_data_2
            pending = [entry for soc, entry in info.get('pending', {}).items()
                       if soc not in records]
            if pending:
                details = '；'.join(f"{entry['soc']}%（{entry['video_sec']:.1f}s）" for entry in pending)
                diagnostics += (f'电量跳变待核对：{details}；时间和容量未写入。'
                                '目标电量可点击表格时刻预览，其余电量可按这里的时刻拖动视频滑块查看。\n')
        self.lbl_status.setText(f"分析完成！1# 提取 {c1} 点（时间待核对 {pending_1} 点），2# 提取 {c2} 点（时间待核对 {pending_2} 点）。")
        QMessageBox.information(
            self, "分析完成",
            f"视频快速分析已完毕！\n"
            f"· 1# 样机捕获: {c1} 个目标节点\n"
            f"· 2# 样机捕获: {c2} 个目标节点\n"
            f"· 时间待核对: 1# {pending_1} 个、2# {pending_2} 个；Excel 对应时间留空，容量仍可导出。\n\n"
            f"{diagnostics}\n"
            f"操作提示:\n"
            f"1. 点击记录可查看识别时原始截图；需要回看视频时点击【定位所选记录的原视频】；\n"
            f"2. 若有识别误差，可直接双击单元格手动修改时间与容量；\n"
            f"3. 确认无误后点击【一键写入 Excel 报告模板】生成报告。"
        )

    def on_worker_error(self, err_msg):
        if self._closing:
            return
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_video.setEnabled(True)
        QMessageBox.critical(self, "分析出错", err_msg)

    def get_table_data_for_export(self, unit_id=1):
        """直接从表格读取数据，100% 以界面显示/用户修改后的值为准"""
        table = self.table_1 if unit_id == 1 else self.table_2
        export_dict = {}
        for row in range(table.rowCount()):
            soc_item = table.item(row, 0)
            time_item = table.item(row, 1)
            cap_item = table.item(row, 2)
            sec_item = table.item(row, 3)

            if not soc_item or not time_item or not cap_item:
                continue

            soc_str = soc_item.text().replace('%', '').strip()
            if not soc_str.isdigit():
                continue
            soc = int(soc_str)

            time_text = time_item.text().strip()
            cap_text = cap_item.text().strip()
            sec_text = sec_item.text().strip() if sec_item else ''

            time_obj = self.parse_time_str(time_text)
            try:
                cap_val = float(cap_text.replace('Ah', '').replace('mAh', '').strip()) if cap_text != '--' else None
            except ValueError:
                cap_val = None
            if time_obj is None and cap_val is None:
                continue

            video_sec = 0.0
            if sec_text and sec_text != '--':
                try:
                    video_sec = float(sec_text.lower().replace('s', '').replace('秒', '').strip())
                except ValueError:
                    video_sec = 0.0

            originals = self.extracted_records_1 if unit_id == 1 else self.extracted_records_2
            original = originals.get(soc, {})
            video_sec = original.get('video_sec', video_sec)
            export_dict[soc] = {
                'soc': soc,
                'time_str': time_text,
                'time_obj': time_obj,
                'capacity': cap_val,
                'video_sec': video_sec
            }
            export_dict[soc].update({key: original[key] for key in LOCATOR_FIELDS if key in original})
        return export_dict

    def export_to_excel(self):
        if not self.project:
            QMessageBox.warning(self, '未选择存档', '请先选择测试存档。')
            return
        pending = set(self.project.metadata.get('planned', [])) - set(self.project.metadata['entries'])
        if pending and QMessageBox.question(self, '计划尚未完成',
                f'还有 {len(pending)} 个计划测试项未完成。仍导出已完成的项目吗？') != QMessageBox.Yes:
            return
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', self.project.metadata['name'])
        default_save = str(self.project.root / (safe_name + '_完整报告.xlsx'))
        save_path, _ = QFileDialog.getSaveFileName(self, '导出完整报告', default_save, 'Excel 文件 (*.xlsx)')
        if not save_path:
            return
        if os.path.normcase(os.path.abspath(save_path)) == os.path.normcase(str(self.project.report_path)):
            QMessageBox.warning(self, '路径无效', '请另存到不同位置，工作版 Excel 需要保留。')
            return
        try:
            count = self.project.export(save_path)
            QMessageBox.information(self, '导出完成', f'已导出 {count} 个测试项：\n{save_path}')
        except Exception as exc:
            QMessageBox.critical(self, '导出失败', str(exc))

    def export_to_csv(self):
        records_1 = self.get_table_data_for_export(1) if self.unit_enabled[1].isChecked() else {}
        records_2 = self.get_table_data_for_export(2) if self.unit_enabled[2].isChecked() else {}

        if not records_1 and not records_2:
            QMessageBox.warning(self, "提示", "没有可导出的数据！")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "导出为 CSV", "battery_test_result.csv", "CSV 文件 (*.csv)")
        if not save_path:
            return

        rows = []
        for unit, records in ((1, records_1), (2, records_2)):
            for soc in get_target_soc_list(self.unit_modes[unit].currentData()):
                if soc in records:
                    rec = records[soc]
                    rows.append({'样机': f'{unit}#', '类型': self.unit_modes[unit].currentText(),
                        '测试项': self.unit_groups[unit].currentText(), '百分比(SOC)': f'{soc}%',
                        '时间': rec.get('time_str', '--'), '容量(Ah)': rec.get('capacity'),
                        '视频时刻(s)': rec.get('video_sec')})

        columns = ['样机', '类型', '测试项', '百分比(SOC)', '时间', '容量(Ah)', '视频时刻(s)']
        with open(save_path, 'w', newline='', encoding='utf-8-sig') as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        QMessageBox.information(self, "成功", f"数据已导出至:\n{save_path}")

    def closeEvent(self, event):
        worker_running = bool(self.worker and self.worker.isRunning())
        monitor = getattr(self, 'resource_monitor', None)
        monitor_running = bool(monitor and monitor.isRunning())
        if worker_running or monitor_running:
            self._closing = True
            if worker_running:
                self.worker.stop()
            if monitor_running:
                monitor.stop()
            self.lbl_status.setText('正在结束后台任务，完成后会自动退出…')
            self._close_timer.start()
            event.ignore()
            return
        self._close_timer.stop()
        if self.video_cap:
            self.video_cap.release()
        super().closeEvent(event)

    def _finish_close(self):
        monitor = getattr(self, 'resource_monitor', None)
        if not (self.worker and self.worker.isRunning()) and not (monitor and monitor.isRunning()):
            self._close_timer.stop()
            self.close()

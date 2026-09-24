# 必须在最顶部导入 onnxruntime，避免 Windows 下 C++ 运行时 DLL 冲突
import onnxruntime
import os
import sys
import cv2
import pandas as pd
import traceback
from datetime import time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QRadioButton,
    QButtonGroup, QSlider, QDoubleSpinBox, QSpinBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox,
    QSplitter, QFrame, QScrollArea, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QColor

from ui.video_canvas import VideoCanvas
from core.video_worker import VideoAnalysisWorker
from core.excel_filler import (
    scan_excel_test_groups, fill_excel_report, get_target_soc_list
)
from core.ocr_engine import OCREngine


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

        self.ocr_tester = None
        self.worker = None

        self.init_ui()
        self.load_excel_groups()

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

        # 测试模式单选
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("测试类型:"))
        self.rb_discharge = QRadioButton("放电测试 (100%→0%)")
        self.rb_charge = QRadioButton("充电测试 (0%→100%)")
        self.rb_discharge.setChecked(True)
        self.rb_discharge.toggled.connect(self.on_mode_changed)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.rb_discharge)
        self.mode_group.addButton(self.rb_charge)
        mode_layout.addWidget(self.rb_discharge)
        mode_layout.addWidget(self.rb_charge)
        grp_video_layout.addLayout(mode_layout)

        # 目标测试项下拉
        grp_video_layout.addWidget(QLabel("选择目标 Excel 测试项:"))
        self.cb_test_group = QComboBox()
        self.cb_test_group.setStyleSheet("height: 28px;")
        grp_video_layout.addWidget(self.cb_test_group)

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


        # 选帧滑块
        grp_roi_layout.addWidget(QLabel("选帧预览 (滑动选择清晰画面):"))
        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setEnabled(False)
        self.slider_frame.valueChanged.connect(self.on_slider_frame_changed)
        grp_roi_layout.addWidget(self.slider_frame)

        self.lbl_frame_info = QLabel("当前帧: 0 / 0 (00:00:00)")
        self.lbl_frame_info.setStyleSheet("color: #666; font-size: 11px;")
        grp_roi_layout.addWidget(self.lbl_frame_info)

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
        self.cb_digit_type.addItems([
            "智能混合模式 (自动尝试多种)",
            "普通数字 (LCD细体/不发光)",
            "断带数码管 (有明显不连贯缝隙)",
            "高亮数码管 (发光太强笔画过粗粘连)"
        ])
        grp_params_layout.addWidget(self.cb_digit_type, 3, 1)

        left_layout.addWidget(grp_params)

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

        export_chk_layout = QHBoxLayout()
        self.chk_export_1 = QCheckBox("同时导出 1# 表格数据")
        self.chk_export_1.setChecked(True)
        self.chk_export_2 = QCheckBox("同时导出 2# 表格数据")
        self.chk_export_2.setChecked(True)
        export_chk_layout.addWidget(self.chk_export_1)
        export_chk_layout.addWidget(self.chk_export_2)
        grp_action_layout.addLayout(export_chk_layout)

        self.btn_export_excel = QPushButton("📊 一键写入 Excel 报告模板")
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
        if not os.path.exists(self.excel_template_path):
            self.lbl_status.setText("提示: 未在当前目录找到默认 Excel 报告模板")
            return

        try:
            self.excel_groups = scan_excel_test_groups(self.excel_template_path)
            self.update_group_combobox()
        except Exception as e:
            QMessageBox.warning(self, "模板加载警告", f"读取 Excel 模板失败: {str(e)}")

    def on_mode_changed(self):
        self.update_group_combobox()
        self.refresh_target_tables()

    def update_group_combobox(self):
        self.cb_test_group.clear()
        sheet_key = "放电时间及曲线" if self.rb_discharge.isChecked() else "充电时间及曲线"

        if sheet_key in self.excel_groups:
            for g in self.excel_groups[sheet_key]:
                self.cb_test_group.addItem(g['name'], g)

    def refresh_target_tables(self):
        """初始化 1# 和 2# 两张 28 个目标行的表格"""
        mode = 'discharge' if self.rb_discharge.isChecked() else 'charge'
        cap_title = "放电容量 (Ah)" if mode == 'discharge' else "充电容量 (Ah)"
        target_socs = get_target_soc_list(mode)

        for unit_id, table in [(1, self.table_1), (2, self.table_2)]:
            table.blockSignals(True)
            table.setHorizontalHeaderLabels([
                "目标电量",
                "时间 (双击修改)",
                f"{cap_title}",
                "发生时刻(点击跳转)",
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

    def select_video(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择录屏视频", "", "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*.*)"
            )
            if not file_path:
                return

            self.video_path = file_path
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
        if not self.video_cap:
            return
        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()
            if ret and frame is not None:
                self.current_frame_cv = frame
                self.canvas.set_frame(frame)
                sec = frame_idx / self.fps
                m, s = divmod(int(sec), 60)
                h, m = divmod(m, 60)
                self.lbl_frame_info.setText(f"当前帧: {frame_idx} / {self.total_frames} ({h:02d}:{m:02d}:{s:02d})")
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")

    def on_roi_changed(self, roi_type, roi):
        names = {
            'soc_1': '1# 电量(蓝)', 'time_1': '1# 时间(绿)', 'cap_1': '1# 容量(橙)',
            'soc_2': '2# 电量(紫)', 'time_2': '2# 时间(青)', 'cap_2': '2# 容量(红)'
        }
        self.lbl_status.setText(f"已更新选区: {names.get(roi_type, roi_type)} {roi}")

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

        if self.video_cap and self.total_frames > 0:
            frame_idx = int(round(video_sec * self.fps))
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(frame_idx)
            self.slider_frame.blockSignals(False)
            self.load_frame(frame_idx)
            soc_text = table.item(row, 0).text() if table.item(row, 0) else ''
            self.lbl_status.setText(f"已跳转预览至 {unit_id}# 样机第 {row+1} 行 (电量 {soc_text}) 对应时刻: {video_sec:.1f}s (第 {frame_idx} 帧)")

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

        tester = self.get_ocr_tester()
        digit_type = self.cb_digit_type.currentIndex()

        if has_1:
            rot_s1 = self.canvas.get_roi_rotation('soc_1')
            rot_t1 = self.canvas.get_roi_rotation('time_1')
            rot_c1 = self.canvas.get_roi_rotation('cap_1')

            c_soc1 = self.crop_helper(self.current_frame_cv, rois['soc_1'], rot_s1)
            c_time1 = self.crop_helper(self.current_frame_cv, rois['time_1'], rot_t1)
            c_cap1 = self.crop_helper(self.current_frame_cv, rois['cap_1'], rot_c1)

            s1, s1_conf = tester.extract_soc(c_soc1, digit_type=digit_type)
            t1_str, _, t1_conf = tester.extract_time(c_time1)
            cap1, raw1, _ = tester.extract_capacity(c_cap1, target_unit=unit_mode)

            rot_desc1 = f" (↻{rot_s1}°已回正)" if rot_s1 != 0 else ""
            msg_lines.append("🟦 1# 样机识别:")
            msg_lines.append(f"  · 电量 (蓝框){rot_desc1}: {s1}% (置信度 {s1_conf:.2f})")
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

            s2, s2_conf = tester.extract_soc(c_soc2, digit_type=digit_type)
            t2_str, _, t2_conf = tester.extract_time(c_time2)
            cap2, raw2, _ = tester.extract_capacity(c_cap2, target_unit=unit_mode)

            rot_desc2 = f" (↻{rot_s2}°已回正)" if rot_s2 != 0 else ""
            msg_lines.append("🟪 2# 样机识别:")
            msg_lines.append(f"  · 电量 (紫框){rot_desc2}: {s2}% (置信度 {s2_conf:.2f})")
            msg_lines.append(f"  · 时间 (青框): {t2_str} (置信度 {t2_conf:.2f})")
            msg_lines.append(f"  · 容量 (红框): {cap2} Ah (原文: '{raw2}')\n")
        else:
            msg_lines.append("🟪 2# 样机: 选区未画满 3 个框 (跳过)\n")

        msg_lines.append("若识别数值准确，即可点击【开始自动快进分析】！")
        QMessageBox.information(self, "单帧测试结果", "\n".join(msg_lines))

    @staticmethod
    def crop_helper(frame, roi, rotation_angle=0):
        if frame is None or roi is None:
            return None
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 1, x))
        y1 = max(0, min(fh - 1, y))
        x2 = max(x1 + 1, min(fw, x + w))
        y2 = max(y1 + 1, min(fh, y + h))
        cropped = frame[y1:y2, x1:x2]

        if rotation_angle and rotation_angle % 360 != 0:
            ang = rotation_angle % 360
            if ang == 90:
                return cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif ang == 180:
                return cv2.rotate(cropped, cv2.ROTATE_180)
            elif ang == 270:
                return cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        return cropped

    def start_analysis(self):
        try:
            if not self.video_path:
                QMessageBox.warning(self, "提示", "请先选择视频文件！")
                return

            rois = self.canvas.rois
            has_1 = bool(rois['soc_1'] and rois['time_1'] and rois['cap_1'])
            has_2 = bool(rois['soc_2'] and rois['time_2'] and rois['cap_2'])

            if not has_1 and not has_2:
                QMessageBox.warning(self, "提示", "请至少为 1# 或 2# 样机画好完整的 3 个选区框！\n· 1# 选区: 蓝、绿、橙框\n· 2# 选区: 紫、青、红框")
                return

            mode = 'discharge' if self.rb_discharge.isChecked() else 'charge'
            interval = self.sp_interval.value()
            debounce = self.sp_debounce.value()
            unit_mode = 'mAh' if 'mAh' in self.cb_cap_unit.currentText() else 'Ah'

            digit_type = self.cb_digit_type.currentIndex()

            self.refresh_target_tables()
            self.extracted_records_1 = {}
            self.extracted_records_2 = {}
            self.extracted_records = self.extracted_records_1

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
                mode=mode,
                sample_interval_sec=interval,
                debounce_count=debounce,
                cap_unit=unit_mode,
                digit_type=digit_type
            )
            self.worker.progress_signal.connect(self.on_worker_progress)
            self.worker.point_detected_signal.connect(self.on_worker_point_detected)
            self.worker.finished_signal.connect(self.on_worker_finished)
            self.worker.error_signal.connect(self.on_worker_error)
            self.worker.start()

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
        self.progress_bar.setValue(pct)
        cur_m, cur_s = divmod(int(cur_sec), 60)
        tot_m, tot_s = divmod(int(total_sec), 60)
        self.lbl_status.setText(
            f"进度: {cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d} ({pct}%) | "
            f"1#:[电量 {soc1_str}, 容量 {cap1_str}] | "
            f"2#:[电量 {soc2_str}, 容量 {cap2_str}]"
        )

    def on_worker_point_detected(self, unit_id, record):
        soc = record['soc']
        table = self.table_1 if unit_id == 1 else self.table_2
        if unit_id == 1:
            self.extracted_records_1[soc] = record
        else:
            self.extracted_records_2[soc] = record

        mode = 'discharge' if self.rb_discharge.isChecked() else 'charge'
        target_socs = get_target_soc_list(mode)
        if soc in target_socs:
            row = target_socs.index(soc)
            table.blockSignals(True)
            table.item(row, 1).setText(record['time_str'])
            table.item(row, 2).setText(f"{record['capacity']:.4f}")
            table.item(row, 3).setText(f"{record['video_sec']:.1f}s")
            item_status = table.item(row, 4)
            item_status.setText("✅ 已捕获")
            item_status.setForeground(QColor(0, 150, 0))
            table.blockSignals(False)

    def on_worker_finished(self, all_data_1, all_data_2):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_video.setEnabled(True)
        self.progress_bar.setValue(100)
        c1 = len(all_data_1)
        c2 = len(all_data_2)
        self.lbl_status.setText(f"分析完成！1# 提取 {c1} 点，2# 提取 {c2} 点。可直接双击修改异常项，或点击写入 Excel。")
        QMessageBox.information(
            self, "分析完成",
            f"视频快速分析已完毕！\n"
            f"· 1# 样机捕获: {c1} 个目标节点\n"
            f"· 2# 样机捕获: {c2} 个目标节点\n\n"
            f"操作提示:\n"
            f"1. 点击任意行的发生时刻可跳转对应视频画面预览；\n"
            f"2. 若有识别误差，可直接双击单元格手动修改时间与容量；\n"
            f"3. 确认无误后点击【一键写入 Excel 报告模板】生成报告。"
        )

    def on_worker_error(self, err_msg):
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

            if not time_text or time_text == '--':
                continue

            time_obj = self.parse_time_str(time_text)
            try:
                cap_val = float(cap_text.replace('Ah', '').replace('mAh', '').strip()) if cap_text != '--' else 0.0
            except ValueError:
                cap_val = 0.0

            video_sec = 0.0
            if sec_text and sec_text != '--':
                try:
                    video_sec = float(sec_text.lower().replace('s', '').replace('秒', '').strip())
                except ValueError:
                    video_sec = 0.0

            export_dict[soc] = {
                'soc': soc,
                'time_str': time_text,
                'time_obj': time_obj if time_obj else time(0, 0, 0),
                'capacity': cap_val,
                'video_sec': video_sec
            }
        return export_dict

    def export_to_excel(self):
        records_1 = self.get_table_data_for_export(unit_id=1) if self.chk_export_1.isChecked() else {}
        records_2 = self.get_table_data_for_export(unit_id=2) if self.chk_export_2.isChecked() else {}

        if not records_1 and not records_2:
            QMessageBox.warning(self, "提示", "您勾选的导出项中没有有效的表格数据。请先提取或勾选有数据的样机。")
            return

        if not os.path.exists(self.excel_template_path):
            file_path, _ = QFileDialog.getOpenFileName(self, "选择 Excel 测试报告模板", "", "Excel 文件 (*.xlsx)")
            if not file_path:
                return
            self.excel_template_path = file_path

        selected_group = self.cb_test_group.currentData()
        if not selected_group:
            QMessageBox.warning(self, "提示", "请先选择要写入的测试项！")
            return

        sheet_name = selected_group['sheet']
        base_row = selected_group['base_row']
        mode = 'discharge' if self.rb_discharge.isChecked() else 'charge'

        default_save = self.excel_template_path.replace(".xlsx", f"_{mode}_{selected_group.get('name_1', selected_group['name'])}.xlsx")
        save_path, _ = QFileDialog.getSaveFileName(self, "另存写入后的 Excel 报告", default_save, "Excel 文件 (*.xlsx)")
        if not save_path:
            return

        try:
            filled_1, filled_2 = fill_excel_report(
                excel_path=self.excel_template_path,
                sheet_name=sheet_name,
                base_row=base_row,
                mode=mode,
                extracted_data=records_1 if records_1 else None,
                output_path=save_path,
                col_offset=0,
                extracted_data_2=records_2 if records_2 else None
            )
            QMessageBox.information(
                self, "写入成功",
                f"🎉 数据已成功写入 Excel 测试报告！\n\n"
                f"📁 保存路径: {save_path}\n"
                f"🟦 1# 写入: {filled_1} 个关键点 (以表格修改后数据为准)\n"
                f"🟪 2# 写入: {filled_2} 个关键点 (以表格修改后数据为准)\n"
                f"🧹 历史残留图片已全部清除\n"
                f"📄 仅保留当前【{sheet_name}】单表\n"
                f"📐 所有原始图表公式已完整保留"
            )
        except Exception as e:
            QMessageBox.critical(self, "写入失败", f"写入 Excel 遇到错误: {str(e)}\n\n{traceback.format_exc()}")

    def export_to_csv(self):
        records_1 = self.get_table_data_for_export(unit_id=1) if self.chk_export_1.isChecked() else {}
        records_2 = self.get_table_data_for_export(unit_id=2) if self.chk_export_2.isChecked() else {}

        if not records_1 and not records_2:
            QMessageBox.warning(self, "提示", "没有可导出的数据！")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "导出为 CSV", "battery_test_result.csv", "CSV 文件 (*.csv)")
        if not save_path:
            return

        mode = 'discharge' if self.rb_discharge.isChecked() else 'charge'
        target_socs = get_target_soc_list(mode)

        rows = []
        for soc in target_socs:
            rec1 = records_1.get(soc, {})
            rec2 = records_2.get(soc, {})
            if rec1 or rec2:
                row_dict = {'百分比(SOC)': f"{soc}%"}
                if records_1:
                    row_dict['1#时间'] = rec1.get('time_str', '--')
                    row_dict['1#容量(Ah)'] = rec1.get('capacity', '--')
                    row_dict['1#视频时刻'] = f"{rec1.get('video_sec', 0.0):.1f}s" if 'video_sec' in rec1 else '--'
                if records_2:
                    row_dict['2#时间'] = rec2.get('time_str', '--')
                    row_dict['2#容量(Ah)'] = rec2.get('capacity', '--')
                    row_dict['2#视频时刻'] = f"{rec2.get('video_sec', 0.0):.1f}s" if 'video_sec' in rec2 else '--'
                rows.append(row_dict)

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        QMessageBox.information(self, "成功", f"数据已导出至:\n{save_path}")

import os
import cv2
import numpy as np
import openpyxl
from datetime import time
from core.ocr_engine import OCREngine
from core.excel_filler import (
    scan_excel_test_groups, fill_excel_report, get_target_soc_list, get_cell_coords
)

def test_ocr_and_excel():
    print("=== 1. 测试 OCR 识别模块 ===")
    ocr = OCREngine()

    # 模拟电量图像
    img_soc = np.ones((60, 100, 3), dtype=np.uint8) * 255
    cv2.putText(img_soc, "99", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    soc_val, soc_conf = ocr.extract_soc(img_soc)
    print(f"电量识别结果: {soc_val}% (置信度: {soc_conf:.2f})")
    assert soc_val == 99, f"期望 99，实际为 {soc_val}"

    # 模拟时间图像
    img_time = np.ones((50, 160, 3), dtype=np.uint8) * 255
    cv2.putText(img_time, "00:02:36", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    time_str, time_obj, time_conf = ocr.extract_time(img_time)
    print(f"时间识别结果: {time_str} (对象: {time_obj})")
    assert time_str == "00:02:36", f"期望 00:02:36，实际为 {time_str}"
    assert time_obj == time(0, 2, 36)

    # 模拟容量图像
    img_cap = np.ones((50, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img_cap, "0.1303 Ah", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cap_val, raw_str, cap_conf = ocr.extract_capacity(img_cap, target_unit='Ah')
    print(f"容量识别结果: {cap_val} Ah (原始文本: '{raw_str}')")
    assert abs(cap_val - 0.1303) < 1e-4, f"期望 0.1303，实际为 {cap_val}"

    print("[OK] OCR 模块测试通过！\n")

    print("=== 2. 测试 Excel 模板清洗、单表导出与双样机(1#+2#)回填 ===")
    template_path = "PB519数显均匀度测试报告9.15（未测完）.xlsx"
    groups = scan_excel_test_groups(template_path)
    print(f"成功扫描到测试工作表: {list(groups.keys())}")
    dis_group = groups['放电时间及曲线'][0]
    print(f"测试填充目标项: {dis_group['name']} (基准行: {dis_group['base_row']})")

    # 构造 1# 和 2# 样机的 28 个目标测试点模拟数据
    target_socs = get_target_soc_list('discharge')
    simulated_data_1 = {}
    simulated_data_2 = {}
    for i, s in enumerate(target_socs):
        simulated_data_1[s] = {
            'time_obj': time(0, (i * 3) // 60, (i * 3) % 60),
            'capacity': round(0.1 + i * 0.15, 4),
            'time_str': f"00:{(i * 3) // 60:02d}:{(i * 3) % 60:02d}",
            'video_sec': i * 180.0
        }
        simulated_data_2[s] = {
            'time_obj': time(0, (i * 3 + 2) // 60, (i * 3 + 2) % 60),
            'capacity': round(0.12 + i * 0.14, 4),
            'time_str': f"00:{(i * 3 + 2) // 60:02d}:{(i * 3 + 2) % 60:02d}",
            'video_sec': i * 180.0 + 10.0
        }

    # 模拟用户在表格上手动修改 1# 的 99%
    simulated_data_1[99]['time_obj'] = time(1, 23, 45)
    simulated_data_1[99]['capacity'] = 8.8888

    test_out_excel = "test_output_report.xlsx"
    res = fill_excel_report(
        excel_path=template_path,
        sheet_name=dis_group['sheet'],
        base_row=dis_group['base_row'],
        mode='discharge',
        extracted_data=simulated_data_1,
        output_path=test_out_excel,
        extracted_data_2=simulated_data_2
    )

    filled_1, filled_2 = res if isinstance(res, tuple) else (res, 0)
    print(f"成功写入关键点数: 1#={filled_1}, 2#={filled_2}")
    assert filled_1 == 28, f"期望 1# 写入 28 点，实际写入 {filled_1}"
    assert filled_2 == 28, f"期望 2# 写入 28 点，实际写入 {filled_2}"

    # 验证新生成的 Excel 文件
    wb_verify = openpyxl.load_workbook(test_out_excel, data_only=False)

    # 1. 验证是否只导出了当前的一个工作表
    print(f"验证工作表列表: {wb_verify.sheetnames}")
    assert wb_verify.sheetnames == ['放电时间及曲线'], f"期望仅导出 ['放电时间及曲线']，实际为 {wb_verify.sheetnames}"

    ws_verify = wb_verify['放电时间及曲线']

    # 2. 验证图片是否已完全删除
    img_count = len(ws_verify._images) if hasattr(ws_verify, '_images') else 0
    print(f"验证图片数量: {img_count}")
    assert img_count == 0, f"期望图片数量为 0，实际为 {img_count}"

    # 3. 验证 1# 写入值 (手动修改生效): 位于第 4 行 Col 2(B), Col 16(P)
    t1_val = ws_verify.cell(4, 2).value
    c1_val = ws_verify.cell(5, 2).value
    print(f"验证 1# 99% 写入值 (手动修改生效): 时间={t1_val}, 容量={c1_val}")
    assert t1_val == time(1, 23, 45), f"期望 1# 时间为 01:23:45，实际为 {t1_val}"
    assert abs(c1_val - 8.8888) < 1e-4, f"期望 1# 容量为 8.8888，实际为 {c1_val}"

    # 4. 验证 2# 写入值: 位于第 4 行 Col 16(P), Col 16(P)
    t2_val = ws_verify.cell(4, 16).value
    c2_val = ws_verify.cell(5, 16).value
    print(f"验证 2# 99% 写入值: 时间={t2_val}, 容量={c2_val}")
    assert t2_val == time(0, 0, 2)
    assert abs(c2_val - 0.12) < 1e-4

    # 5. 检查 1# 和 2# 公式是否依然完好保留
    f1_1 = ws_verify.cell(4, 12).value
    f1_2 = ws_verify.cell(4, 13).value
    f2_1 = ws_verify.cell(4, 26).value
    f2_2 = ws_verify.cell(4, 27).value
    print(f"验证 1# 公式保留: 每10%='{f1_1}', 占比='{f1_2}'")
    print(f"验证 2# 公式保留: 每10%='{f2_1}', 占比='{f2_2}'")
    assert f1_1 == '=K4-B4'
    assert f1_2 == '=L4/K31'
    assert f2_1 == '=Y4-P4'
    assert f2_2 == '=Z4/Y31'

    print("[OK] Excel 模板清洗、单表导出、双样机(1#+2#)回填、手动修改生效与公式保留全部测试通过！\n")

    # 清理临时文件
    if os.path.exists(test_out_excel):
        os.remove(test_out_excel)
        print("[OK] 测试临时文件已自动清理。")

if __name__ == '__main__':
    test_ocr_and_excel()

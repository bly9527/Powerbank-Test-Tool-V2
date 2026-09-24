import sys, os, glob
with open("test_glob.txt", "w", encoding="utf-8") as f:
    f.write(f"frozen: {getattr(sys, 'frozen', False)}\n")
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    f.write(f"base_path: {base_path}\n")
    f.write(f"files in base_path: {os.listdir(base_path)}\n")
    f.write(f"xlsx files: {glob.glob(os.path.join(base_path, '*.xlsx'))}\n")

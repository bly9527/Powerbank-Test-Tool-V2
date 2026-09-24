import os
import shutil

src_dir = r"C:\Users\30583\Desktop\充电宝数显均匀度自动测试工具\dist\PB_Test_Tool_v2"
dest_dir = r"C:\Users\30583\Desktop\新建文件夹 (4)\充电宝数显均匀度自动测试工具v2"

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

print(f"Moving {src_dir} to {dest_dir}")
shutil.move(src_dir, dest_dir)
print("Moved successfully!")

import os, win32com.client

desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
shortcut_path = os.path.join(desktop, '充电宝数显测试工具v2(全功能).lnk')

# target is inside dist\PB_Test_Tool_v2
target_path = os.path.join(os.getcwd(), 'dist', 'PB_Test_Tool_v2', 'PB_Test_Tool_v2.exe')
working_dir = os.path.join(os.getcwd(), 'dist', 'PB_Test_Tool_v2')

shell = win32com.client.Dispatch('WScript.Shell')
shortcut = shell.CreateShortCut(shortcut_path)
shortcut.Targetpath = target_path
shortcut.WorkingDirectory = working_dir
shortcut.IconLocation = target_path
shortcut.save()
print(f"Created shortcut at {shortcut_path}")

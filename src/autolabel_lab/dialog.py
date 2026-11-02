"""Small native file picker, isolated from the web server's threads."""
import json
import sys
import tkinter as tk
from tkinter import filedialog


def main():
    kind = sys.argv[1]
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if kind == "directory":
        selected = filedialog.askdirectory(title="选择已有实验目录", parent=root)
    else:
        filters = [("视频", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("所有文件", "*.*")] if kind == "video" else [("应用程序", "*.exe"), ("所有文件", "*.*")]
        selected = filedialog.askopenfilename(title="选择视频" if kind == "video" else "选择 X-AnyLabeling 程序", filetypes=filters, parent=root)
    root.destroy()
    print(json.dumps({"path": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()

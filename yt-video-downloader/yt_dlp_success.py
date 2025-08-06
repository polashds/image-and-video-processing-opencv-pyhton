import tkinter as tk
from tkinter import messagebox
import yt_dlp
import os

def download_video():
    url = url_entry.get()
    if not url:
        messagebox.showerror("Error", "Please enter a YouTube URL!")
        return

    try:
        ydl_opts = {
            'outtmpl': os.path.join(os.path.expanduser('~'), 'Downloads', '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'video')
            ydl.download([url])
            messagebox.showinfo("Success", f"Downloaded: {video_title}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to download: {str(e)}")

# GUI Setup
root = tk.Tk()
root.title("YouTube Downloader (yt-dlp)")

# URL Entry
tk.Label(root, text="YouTube URL:").pack(pady=5)
url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

# Download Button
download_btn = tk.Button(root, text="Download", command=download_video)
download_btn.pack(pady=10)

root.mainloop()
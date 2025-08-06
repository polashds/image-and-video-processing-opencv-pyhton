import tkinter as tk
from tkinter import messagebox
from pytube import YouTube
import os

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

def Downloader():
    url = entry.get()
    if not url:
        messagebox.showerror("Error", "Please enter a URL")
        return
    
    try:
        yt = YouTube(url)
        video = yt.streams.get_highest_resolution()
        download_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        video.download(output_path=download_path)
        messagebox.showinfo("Success", "Video downloaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to download video: {str(e)}")

# Main window setup
root = tk.Tk()
root.title("YouTube Downloader")

# This will handle window close button properly
root.protocol("WM_DELETE_WINDOW", on_closing)

# Your existing UI elements
label = tk.Label(root, text="Enter YouTube URL:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

button = tk.Button(root, text="Download", command=Downloader)
button.pack()

root.mainloop()
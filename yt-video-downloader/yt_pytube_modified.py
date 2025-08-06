from pytube import YouTube
import tkinter as tk
from tkinter import messagebox
import os

def Downloader():
    url = entry.get()
    if not url:
        messagebox.showerror("Error", "Please enter a URL")
        return
    
    try:
        yt = YouTube(url)
        # Get the highest resolution stream
        video = yt.streams.get_highest_resolution()
        # Or for audio: yt.streams.get_audio_only()
        
        # Set output path (optional)
        download_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        
        # Download the video
        video.download(output_path=download_path)
        messagebox.showinfo("Success", "Video downloaded successfully!")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to download video: {str(e)}")

# Your existing Tkinter setup
root = tk.Tk()
root.title("YouTube Downloader")

label = tk.Label(root, text="Enter YouTube URL:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

button = tk.Button(root, text="Download", command=Downloader)
button.pack()

root.mainloop()
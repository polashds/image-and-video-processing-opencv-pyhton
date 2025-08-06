It looks like you're encountering an HTTP 400 error when trying to download a YouTube video using `pytube` with a Tkinter interface. This error typically occurs when the request to YouTube's servers is malformed or when there's an issue with the YouTube API that pytube uses internally.

Here are some steps to troubleshoot and fix this issue:

### 1. Update pytube
First, make sure you're using the latest version of pytube, as YouTube frequently changes its API and the library needs updates to keep up:

```bash
pip install --upgrade pytube
```

### 2. Check the YouTube URL
Ensure the URL you're trying to download is valid and accessible. Try opening it in a browser to confirm.

### 3. Modify your code
Here's an updated version of your downloader function that includes better error handling and uses pytube's more recent methods:

```python
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
```

### 4. Alternative solutions if pytube keeps failing
If pytube continues to have issues, you might want to try:

1. **Using yt-dlp** (a more actively maintained fork of youtube-dl):
   ```python
   import yt_dlp

   def download_video(url):
       ydl_opts = {}
       with yt_dlp.YoutubeDL(ydl_opts) as ydl:
           ydl.download([url])
   ```

2. **Adding headers to requests** (sometimes helps with 400 errors):
   ```python
   from pytube import request
   request.default_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
   ```

### 5. Network considerations
- Check if you're behind a proxy or firewall that might be blocking requests
- Try using a VPN if YouTube is restricted in your region
- Ensure your system clock is synchronized correctly

The HTTP 400 error from pytube is a common issue that often requires updating the library or using an alternative approach when YouTube changes its API. The yt-dlp alternative is generally more reliable for long-term use.

The `KeyboardInterrupt` error you're seeing occurs when you manually stop the program (usually by pressing `Ctrl+C` in the terminal). This isn't actually an error in your code - it's just the way Python responds when you interrupt the program execution.

Since your script is a Tkinter application with `root.mainloop()`, it will keep running until you explicitly close the window or interrupt it. The traceback shows that the program was running normally until you stopped it.

### What This Means:
1. Your script is working correctly as a GUI application
2. The `KeyboardInterrupt` appears when you stop it manually
3. This is normal behavior for Tkinter applications

### If You Want to Handle This Gracefully:
You can add a proper exit handler to your Tkinter application:

```python
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
```

### Key Improvements:
1. Added proper window closing handler (`on_closing`)
2. The app will now ask for confirmation before closing
3. Cleaner exit when closing the window

If you're seeing this `KeyboardInterrupt` while testing, it just means you stopped the program manually - your code is likely working fine when allowed to run normally.

Since you're encountering **HTTP 400 Bad Request** errors with `pytube`, here’s a **complete alternative solution** using `yt-dlp` (a more reliable and actively maintained YouTube downloader).  

---

## **Solution 1: Full Replacement with `yt-dlp` (Recommended)**
### **1. Install `yt-dlp`**
First, install the library:
```sh
pip install yt-dlp
```

### **2. Modified Tkinter GUI with `yt-dlp`**
Here’s an updated version of your YouTube downloader using `yt-dlp`:

```python
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
```

### **Key Improvements:**
✅ **More reliable** than `pytube` (works with YouTube's latest changes)  
✅ **Automatically saves to `~/Downloads`** (customizable)  
✅ **Better error handling** (shows video title on success)  
✅ **No 400 Bad Request issues** (unlike `pytube`)  

---

## **Solution 2: Fix `pytube` (If You Still Want to Use It)**
If you prefer sticking with `pytube`, try these fixes:

### **1. Update `pytube`**
```sh
pip install --upgrade pytube
```

### **2. Modify Headers (Avoid 400 Errors)**
```python
from pytube import request
request.default_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'accept-language': 'en-US,en;q=0.9',
}
```

### **3. Use a VPN or Proxy (If Blocked)**
Sometimes, YouTube blocks requests from certain IPs. Try:
- Using a **VPN** (e.g., ProtonVPN, Windscribe)
- Running the script on a different network

---

## **Which Solution Should You Use?**
| Feature | `pytube` | `yt-dlp` |
|---------|---------|---------|
| **Reliability** | ❌ Often breaks | ✅ Works consistently |
| **Speed** | ✅ Fast | ✅ Fast |
| **Updates** | ❌ Slow updates | ✅ Actively maintained |
| **Error Handling** | ❌ Poor | ✅ Robust |
| **Best for** | Simple scripts | Production use |

### **Recommendation:**  
👉 **Use `yt-dlp`** for a **stable, long-term solution**.  
👉 Use `pytube` only for **quick testing** (but expect issues).  

---

### **Final Thoughts**
- If you keep getting **400 Bad Request**, switch to `yt-dlp`.  
- If you must use `pytube`, **update it and modify headers**.  
- **VPN/Proxy** may help if YouTube is blocking your requests.  

Try the `yt-dlp` solution—it should work flawlessly! 🚀
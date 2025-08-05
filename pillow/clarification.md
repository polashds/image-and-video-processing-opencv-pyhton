You're getting the error:

```
AttributeError: 'str' object has no attribute 'size'
```

because you're trying to access `.size` and `.mode` on the **file path string**, not the actual image object.

---

### ✅ Fix

You should call `.size` and `.mode` on the **`Image` object**, not the **string path**.

Assuming you already opened the image using this:

```python
lena_image_pil = Image.open(lena_image_path)
```

Then do this:

```python
# Get image size
lena_image_size = lena_image_pil.size

# Get image mode
lena_image_mode = lena_image_pil.mode

# Print results
print("lena_image_size class:", type(lena_image_size))
print("lena_image_size length:", len(lena_image_size))
print("Lena image size:", lena_image_size)
print()
print("lena_image_mode class:", type(lena_image_mode))
print("Lena image mode:", lena_image_mode)
```

---

### 🔍 Explanation

* `.size` → returns a tuple `(width, height)`
* `.mode` → returns a string like `'RGB'`, `'L'` (grayscale), etc.

Let me know if you want to convert it to grayscale or NumPy array next.


Great catch — you're almost there! The problem is subtle but important:

---

### ❌ What's Wrong?

You're doing this:

```python
data_dir = os.path.join("E:", "CV-Projects-Vault", ...)
```

Which produces this:

```
E:CV-Projects-Vault\image-and-video-processing-py-cv\resources\lena.png
```

That is **not** an absolute path. On Windows, `"E:"` (without a backslash) means “current directory on drive E”, **not the root of E:**. That's why the file isn’t found.

---

### ✅ The Fix

Use `"E:\\"` instead of `"E:"` so it knows you mean **root of drive E**:

```python
import os
from PIL import Image

# Build the correct absolute path using os.path.join
data_dir = os.path.join("E:\\", "CV-Projects-Vault", "image-and-video-processing-py-cv")
resource_dir = os.path.join(data_dir, "resources")
lena_image_path = os.path.join(resource_dir, "lena.png")

# Debug prints
print("data_dir class:", type(data_dir))
print("Data directory:", data_dir)
print()
print("resource_dir class:", type(resource_dir))
print("Resource directory:", resource_dir)
print()
print("lena_image_path:", lena_image_path)
print("File exists:", os.path.exists(lena_image_path))  # Should return True now

# Load and show the image
lena_image_pil = Image.open(lena_image_path)
print("lena_image_pil type:", type(lena_image_pil))
lena_image_pil.show()
```

---

### 🧠 Key Difference

| `"E:"`   | Relative to current directory on `E:` |
| -------- | ------------------------------------- |
| `"E:\\"` | Absolute path from root of `E:`       |

---

Let me know if you want to make this script auto-detect drives or work across OSes.


You're seeing:

```
cannot convert --f=c:\Users\surface\AppData\Roaming\jupyter\runtime\kernel-....json
```

because you're running this code **inside a Jupyter Notebook**, and `sys.argv` in that environment includes the kernel's config file (used internally by Jupyter to manage sessions). That `.json` file is not an image, hence the `OSError`.

---

### ✅ Why this happens

In a **Jupyter Notebook**, when you do:

```python
for infile in sys.argv[1:]:
```

You get:

```python
['--f=c:\\Users\\...\\kernel-xyz.json']
```

Because Jupyter passes that kernel file as a command-line argument. But this code is meant to be run in a **Python script** from the **command line**, like:

```bash
python convert_to_jpg.py image1.png image2.png
```

---

### ✅ Fix: Make the code Jupyter-safe

If you're testing this inside Jupyter, avoid using `sys.argv`. Instead, manually pass your filenames:

```python
import os
from PIL import Image

# List of files to convert
input_files = [
    r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lena.png",
    r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lambo.png"
]

# Convert to JPG
for infile in input_files:
    f, e = os.path.splitext(infile)
    outfile = f + ".jpg"
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                rgb_im = im.convert('RGB')  # Ensure compatibility with JPEG
                rgb_im.save(outfile, "JPEG")
                print("Saved:", outfile)
        except OSError as e:
            print("Cannot convert", infile, "| Error:", e)
```

---

### ✅ If You Want to Keep `sys.argv` in a `.py` Script

Save this as `convert_to_jpg.py`, then run it from the terminal like:

```bash
python convert_to_jpg.py lena.png lambo.png
```

In that context, `sys.argv[1:]` will work as expected.

---

Let me know if you want to auto-scan a folder for images and convert all to `.jpg`.


You're getting **no output in Jupyter Notebook** because `sys.argv[1:]` is empty or contains only non-image arguments like the kernel file.

---

### 🔍 What’s Happening?

In Jupyter, this:

```python
import sys
print(sys.argv)
```

typically prints:

```
['/path/to/kernel-xyz.json']
```

So when you run:

```python
for infile in sys.argv[1:]:
```

It:

* **skips non-image inputs**
* **or silently fails** (due to your `except OSError: pass`)

---

### ✅ Solution for Jupyter

In Jupyter, you should **pass a list of image paths manually**.

#### ✅ Modified Version for Jupyter Notebook:

```python
from PIL import Image

# Manually provide image file paths
input_files = [
    r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lena.png",
    r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lambo.png"
]

for infile in input_files:
    try:
        with Image.open(infile) as im:
            print(infile, im.format, f"{im.size} {im.mode}")
    except OSError as e:
        print(f"Cannot open {infile}: {e}")
```

---

### ✅ If You Want It to Work as a `.py` Script

Save it like this:

```python
# save as inspect_images.py
import sys
from PIL import Image

for infile in sys.argv[1:]:
    try:
        with Image.open(infile) as im:
            print(infile, im.format, f"{im.size} {im.mode}")
    except OSError:
        print("Cannot open", infile)
```

Then run from the terminal:

```bash
python inspect_images.py lena.png lambo.png
```

---

Let me know if you'd like to:

* Batch process all images in a folder
* Filter by file extension
* Convert formats automatically


You're asking to use the `merge()` function for images in a **Jupyter Notebook**, just like in a regular Python script. Your function is already correctly defined.

Here’s a **full Jupyter-friendly version**, where we:

1. Load two images manually (instead of using command-line arguments).
2. Call your `merge()` function.
3. Display the result inline using `IPython.display`.

---

### ✅ Full Working Code in Jupyter Notebook

```python
from PIL import Image
from IPython.display import display

def merge(im1: Image.Image, im2: Image.Image) -> Image.Image:
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGBA", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im

# Load images manually (make sure the paths are correct)
im1_path = r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lena.png"
im2_path = r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lambo.png"

im1 = Image.open(im1_path).convert("RGBA")  # Ensure same mode
im2 = Image.open(im2_path).convert("RGBA")

# Merge and display
merged_image = merge(im1, im2)
display(merged_image)
```

---

### 📌 Notes:

* `.convert("RGBA")` ensures both images have the same mode.
* `display()` from `IPython.display` shows the image **inline in Jupyter**.
* If the images differ significantly in height, the shorter one will have transparent padding.

---

Let me know if you want to:

* Merge images vertically instead of horizontally
* Save the merged image
* Add a margin or spacing between them


Absolutely! Here's a **Jupyter Notebook–friendly version** of your `ImageOps` resizing example, but using your **`lena.png`** image instead of `hopper.webp`.

---

### ✅ Jupyter Notebook Code for `lena.png`

```python
from PIL import Image, ImageOps
from IPython.display import display
import os

# Define the image size target
size = (100, 150)

# Path to your image (adjust if needed)
lena_path = r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lena.png"

# Load the image and apply ImageOps transformations
with Image.open(lena_path) as im:
    # Contain
    contained = ImageOps.contain(im, size)
    print("ImageOps.contain:")
    display(contained)

    # Cover
    covered = ImageOps.cover(im, size)
    print("ImageOps.cover:")
    display(covered)

    # Fit (center crop)
    fitted = ImageOps.fit(im, size)
    print("ImageOps.fit:")
    display(fitted)

    # Pad with red background
    padded = ImageOps.pad(im, size, color="#f00")
    print("ImageOps.pad (red):")
    display(padded)

    # Thumbnail (in-place)
    thumb = im.copy()
    thumb.thumbnail(size)
    print("Image.thumbnail:")
    display(thumb)
```

---

### 📝 Notes

* You **must update** the `lena_path` if your file is in a different location.
* `display(...)` shows the image below the print title in the output cell.
* `thumbnail()` changes the image in place, so we use `im.copy()` to avoid altering the original `im`.

---

Let me know if you want to:

* Add labels below the images
* Display them side-by-side in a grid
* Save them as `.jpg` or `.webp` files after transformation
To make your `compress_image` function work in **Jupyter Notebook**, we’ll:

1. Avoid `glob.glob("*.png")` (since Jupyter's working directory might not be where your images are).
2. Instead, either:

   * Provide a full directory path, or
   * List image filenames manually.

I'll show both options below 👇

---

## ✅ Option 1: Jupyter-Friendly — Manually List PNGs

```python
from PIL import Image

def compress_image(source_path: str, dest_path: str) -> None:
    with Image.open(source_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dest_path, "JPEG", optimize=True, quality=80)

# Manually specify PNG image paths
image_paths = [
    r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lena.png",
    r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\lambo.png"
]

# Compress each to JPEG
for path in image_paths:
    output_path = path[:-4] + ".jpg"  # Replace .png with .jpg
    compress_image(path, output_path)
    print(f"Compressed and saved: {output_path}")
```

---

## ✅ Option 2: Automatically Scan a Directory for `.png` Files

```python
import glob
import os
from PIL import Image

def compress_image(source_path: str, dest_path: str) -> None:
    with Image.open(source_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dest_path, "JPEG", optimize=True, quality=80)

# Define the folder where your .png images are
folder_path = r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources"

# Get all .png files in that folder
png_paths = glob.glob(os.path.join(folder_path, "*.png"))

# Compress each image to JPEG
for path in png_paths:
    output_path = path[:-4] + ".jpg"
    compress_image(path, output_path)
    print(f"Compressed: {output_path}")
```

---

### ✅ Bonus Tip: Show the Compressed Image Inline (Jupyter Only)

```python
from IPython.display import display

display(Image.open(output_path))
```

---

Let me know if you'd like to compress all images in **subfolders** too, or control the quality dynamically.


Perfect! You're using `pathlib`, which is cleaner and more modern than `glob` or `os`. Here's how to adapt your code to run safely in **Jupyter Notebook**, with optional full-path support, progress printing, and inline image display.

---

### ✅ Jupyter-Friendly Version using `pathlib.Path`

```python
from pathlib import Path
from PIL import Image
from IPython.display import display

def compress_image(source_path: Path, dest_path: Path) -> None:
    with Image.open(source_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dest_path, "JPEG", optimize=True, quality=80)

# Option 1: Use current directory (Jupyter's working dir)
# folder = Path(".")

# ✅ Option 2 (Recommended): Use full path to your images folder
folder = Path(r"E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources")

# Iterate through .png files
paths = folder.glob("*.png")

for path in paths:
    output_path = path.with_suffix(".jpg")  # replaces .png with .jpg
    compress_image(path, output_path)
    print(f"Compressed: {output_path.name}")
    display(Image.open(output_path))
```

---

### 📝 Notes:

* `path.stem + ".jpg"` gives just the filename, but you’ll need to resolve the **full path** to write the file. So better to use `path.with_suffix(".jpg")`.
* `display()` from `IPython.display` shows each JPEG inline.
* Works even if you're inside a nested Jupyter working directory.

---

Let me know if you want to:

* Save outputs to a separate folder
* Add resizing along with compression
* Create a before-vs-after comparison grid

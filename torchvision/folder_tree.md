In Windows Jupyter Notebook, you can show the folder tree in several ways.
Here are three common methods:

---

**1️⃣ Using `os.walk()` (pure Python)**

```python
import os

def show_tree(start_path, indent=""):
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, "").count(os.sep)
        indent_str = "    " * level
        print(f"{indent_str}{os.path.basename(root)}/")
        sub_indent = "    " * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

# Example usage
show_tree(r"C:\path\to\your\folder")
```

---

**2️⃣ Using `tree` command (Windows shell)**
In Jupyter, prefix shell commands with `!`:

```python
!tree /F "C:\path\to\your\folder"
```

* `/F` → lists files as well as folders.
* Remove `/F` if you want only folder names.

---

**3️⃣ Using `IPython.display` for nicer formatting**

```python
from IPython.display import Markdown
import os

def tree_md(start_path):
    md = []
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, "").count(os.sep)
        indent = "&nbsp;&nbsp;&nbsp;" * level
        md.append(f"{indent}📂 **{os.path.basename(root)}/**")
        for f in files:
            md.append(f"{indent}&nbsp;&nbsp;&nbsp;📄 {f}")
    return Markdown("\n".join(md))

tree_md(r"C:\path\to\your\folder")
```

This shows the directory tree in a more **Jupyter-friendly styled** way.

---

If you want, I can make you a **`show_tree()` function that works like the Linux `tree` command** but runs entirely in Python so it works even if you don't have the `tree` command installed. That’s best for portability.

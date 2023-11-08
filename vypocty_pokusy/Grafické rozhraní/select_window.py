import tkinter as tk
from tkinter import filedialog
import os


def browse_files(window_title="Vyberte soubory"):
    file_paths = filedialog.askopenfilenames(
        title=window_title,
        filetypes=(("Obrázkové soubory (CV2)", "*.jpg *.jpeg *.jpe *.JPG *.jp2 *.png *.bmp *.dib *.webp *.avif *.pbm"
                                               "*.pgm *.ppm *.pxm *.pnm *.pfm *.sr *.ras *.tiff *.tif *.exr *.hdr *.pic"
                    ),
                   ("Obrázkové soubory (Video)", "*.jpg *.jpeg *.JPG *.png *.tif *.tiff *.webp"),
                   ("Video soubory", "*.mp4 *.avi"),
                   ("Textové soubory", "*.txt"),
                   ("ZIP soubory", "*.zip"),
                   ("PDF soubory", "*.pdf"),
                   ("JSON soubory", "*.json"),
                   ("YAML soubory", "*.yaml"),
                   ("YAML soubory", "*.yaml"),
                   ("Všechny soubory", "*.*"))
    )
    """for file_path in file_paths:
        # Zde můžete provádět operace se soubory (např. zobrazení názvu souboru).
        print("Vybrán soubor:", file_path)"""
    file_paths = [os.path.basename(os.path.splitext(file)[0]) for file in file_paths]
    return file_paths


def browse_directory(window_title="Vyberte složku"):
    folder_path = []
    while True:
        directory = filedialog.askdirectory(title=window_title)
        if not directory:  # Uživatel klikl na "Cancel" nebo zavřel dialog
            break
        folder_path.append(directory)
    """if folder_path:
        print("Vybraná složka:", folder_path)"""
    return folder_path

browse_directory()
"""root = tk.Tk()
root.title("Označení souborů")

browse_button = tk.Button(root, text="Vyberte soubory", command=browse_files)
browse_button.pack()
browse_button = tk.Button(root, text="Vyberte složku", command=browse_directory)
browse_button.pack()

root.mainloop()"""

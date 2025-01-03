import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
import h5py
import numpy as np


# Function to select a file
def select_file(entry):
    path = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat")])
    if path:
        entry.delete(0, ctk.END)
        entry.insert(0, path)


# Function to select a folder
def select_folder(entry):
    path = filedialog.askdirectory()
    if path:
        entry.delete(0, ctk.END)
        entry.insert(0, path)


# Function to update paths
def update_paths(data, new_base_path):
    for column in data.keys():
        if isinstance(data[column], list):
            for i, path in enumerate(data[column]):
                if isinstance(path, str):
                    file_name = os.path.basename(path)
                    data[column][i] = os.path.join(new_base_path, file_name)
    return data


# Main function to process files
def process_files():
    input_path = input_entry.get()
    output_path = output_entry.get()
    process_all = process_all_var.get()

    if not os.path.exists(input_path):
        messagebox.showerror("Error", "Input path does not exist!")
        return
    if not os.path.exists(output_path) or not os.path.isdir(output_path):
        messagebox.showerror("Error", "Output path does not exist!")
        return

    if process_all and not os.path.isdir(input_path):
        messagebox.showerror("Error", "Input path is not a folder!")
        return
    else:
        if input_path == output_path:
            messagebox.showerror("Error", "Input and output paths cannot be the same!")
            return

    if not process_all and os.path.isdir(input_path):
        messagebox.showerror("Error", "Input path is a folder, but 'Process all files' is not checked!")
        return
    else:
        if os.path.dirname(input_path) == output_path:
            messagebox.showerror("Error", "Input and output paths cannot be the same!")
            return

    # try:
    if process_all:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".mat"):
                    input_file = os.path.join(root, file)
                    subfolder = os.path.basename(os.path.dirname(input_file))
                    new_folder = os.path.join(output_path, subfolder)
                    # output_path = os.path.join(new_folder, os.path.basename(input_path))
                    os.makedirs(new_folder, exist_ok=True)
                    process_single_file(input_file, output_path)
    else:
        result = process_single_file(input_path, output_path)
        if result:
            messagebox.showinfo("Done", "Processing completed!")
    # except Exception as e:
    #     messagebox.showerror("Error", f"An error occurred: {e}")


# Function to process a single file
def process_single_file(input_file, output_folder):
    # Načtení .mat souboru
    with h5py.File(input_file, 'r+') as file:
        file_keys = list(file.keys())

        if 'plottingData' in file_keys:
            messagebox.showerror("Error", "Please select Ncorr .mat file!\n\tThis file is not a Ncorr_POST file.")
            return False

        # reference_img_path = file['reference_save']['path'][:].tobytes().decode('utf-16')

        if 'reference_save' in file and 'path' in file['reference_save']:
            del file['reference_save']['path']

        new_path = np.array([ord(char) for char in str(output_folder)], dtype=np.uint16)[:, np.newaxis]

        # Vytvoření nového datasetu 'path' s chunkingem
        file.create_dataset('reference_save/path', shape=(len(new_path), 1), chunks=(1, 1),
                                data=new_path, dtype=np.uint16)

        # normal_img_paths = [file[file['current_save']['path'][i][0]][:].tobytes().decode('utf-16') for i in range(file['current_save']['path'].size)]

        for i in range(file['current_save']['path'].size):
            file[file['current_save']['path'][i][0]][:] = new_path

        if 'current_save' in file and 'path' in file['current_save']:
            del file['current_save']['path']

        new_paths = np.array([[ord(char) for char in str(output_folder)], [ord(char) for char in str(output_folder)]], dtype=np.uint16)[:, :, np.newaxis]
        file.create_dataset('current_save/path', data=new_paths)


    file.close()

    # output_path = os.path.join(output_folder, os.path.basename(input_file))
    # with h5py.File(output_path, 'w') as new_file:
    #     # Uložení dat
    #     for file_key in file_keys:
    #         new_file.create_dataset(file_key, data=file_new[file_key])
    # new_file.close()
    return True


# GUI setup
app = ctk.CTk()
app.title("MAT File Path Editor")
app.geometry("380x160")
app.resizable(False, False)

# Configure grid to center contents
app.grid_rowconfigure(0, weight=1)
app.grid_rowconfigure(1, weight=1)
app.grid_rowconfigure(2, weight=1)
app.grid_rowconfigure(3, weight=1)
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)
app.grid_columnconfigure(2, weight=1)

# Input path
ctk.CTkLabel(app, text="Input Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
input_entry = ctk.CTkEntry(app, width=230)
input_entry.grid(row=0, column=1, padx=5, pady=5)
ctk.CTkButton(app, text="...", width=20, command=lambda: select_file(input_entry)).grid(row=0, column=2, padx=5, pady=5)

# Output path
ctk.CTkLabel(app, text="Output Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
output_entry = ctk.CTkEntry(app, width=230)
output_entry.grid(row=1, column=1, padx=5, pady=5)
ctk.CTkButton(app, text="...", width=20, command=lambda: select_folder(output_entry)).grid(row=1, column=2, padx=5,
                                                                                           pady=5)

# Checkbox for processing all files
process_all_var = ctk.BooleanVar()
ctk.CTkCheckBox(app, border_width=2, corner_radius=8, text="Process all files in folder",
                variable=process_all_var).grid(row=2, column=1, pady=5)

# Button to start processing
ctk.CTkButton(app, text="Start Processing", command=process_files).grid(row=3, column=1, pady=10)

app.mainloop()

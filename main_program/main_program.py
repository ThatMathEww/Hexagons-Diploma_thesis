import os
import sys
import cv2
import time
import ctypes
import numpy as np
import matplotlib.pyplot as plt


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        obj_without_color = self.remove_color_escape_sequences(obj)

        for f in self.files:
            try:
                f.write(obj_without_color)
                f.flush()  # Zápis a flush mohou být provedeny současně
            except (AttributeError, ValueError):
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except (AttributeError, ValueError):
                pass

    @staticmethod
    def remove_color_escape_sequences(text):
        in_escape_sequence = False
        filtered_text = ''

        for char in text:
            if char == '\033':
                in_escape_sequence = True
            elif in_escape_sequence and char == 'm':
                in_escape_sequence = False
            elif not in_escape_sequence:
                filtered_text += char

        return filtered_text


"""class GlobalImport:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        import inspect
        collector = inspect.getargvalues(inspect.getouterframes(inspect.currentframe())[1][0]).locals
        globals().update(collector)"""


class SaveError(Exception):
    def __init__(self, error):
        self.error = error
        super().__init__(self.error)


class MyException(Exception):
    def __init__(self, error):
        self.error = error
        super().__init__(self.error)


def get_photos_from_folder(folder, file_list=None, img_types=(".jpg", ".jpeg", ".JPG", ".png", ".tiff", ".tif")):
    if any(item not in (".jpg", ".jpeg", ".jpe", ".JPG", ".jp2", ".png", ".bmp", ".dib", ".webp",
                        ".avif", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".pfm", ".sr", ".ras",
                        ".tiff", ".tif", ".exr", ".hdr", ".pic") for item in img_types):
        program_shutdown(f"\n\033[31;1;21mERROR\033[0m"
                         f"\n\tNepodporovaný typ fotografie.\n\t\033[41;30m➤ Ukončení programu.\033[0m",
                         try_save=False)
    else:
        files = [f for f in (os.listdir(folder) if file_list is None else file_list)
                 if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(img_types)]
        if not files:
            print("\n\033[1;21mWARRNING\033[0m\n\tSložka je prázdná.")
            return files
        first_type = os.path.splitext(files[0])[1]
        if all(os.path.splitext(f)[1] == first_type for f in files):  # kontrola jestli josu všechny fotky stejné
            return files
        else:
            print("\n\033[33;1;21mWARRNING\033[0m\n\tNačtené fotografie jsou různého typu.")
            return None


def browse_files(window_title="Vyberte soubory"):
    while True:
        file_paths = filedialog.askopenfilenames(
            title=window_title,
            filetypes=(
                ("Obrázkové soubory (CV2)", "*.jpg *.jpeg *.jpe *.JPG *.jp2 *.png *.bmp *.dib *.webp *.avif *.pbm"
                                            "*.pgm *.ppm *.pxm *.pnm *.pfm *.sr *.ras *.tiff *.tif *.exr *.hdr *.pic"),
                ("Obrázkové soubory (Video)", "*.jpg *.jpeg *.JPG *.png *.tif *.tiff *.webp"),
                ("Video soubory", "*.mp4 *.avi"),
                ("Textové soubory", "*.txt"),
                ("ZIP soubory", "*.zip"),
                ("PDF soubory", "*.pdf"),
                ("JSON soubory", "*.json"),
                ("YAML soubory", "*.yaml"),
                ("YAML soubory", "*.yaml"),
                ("Všechny soubory", "*.*")))
        if len(file_paths) > 0:
            break
        else:
            print("\n\tNebyly vybrány žádné soubory.")
    """for file_path in file_paths:
                # Zde můžete provádět operace se soubory (např. zobrazení názvu souboru).
                print("Vybrán soubor:", file_path)"""
    file_paths = [os.path.basename(os.path.splitext(file)[0]) for file in file_paths]
    return file_paths


def browse_directory(window_title="Vyberte složku"):
    while True:
        folder_path = []
        while True:
            directory = filedialog.askdirectory(title=window_title)
            if not directory:  # Uživatel klikl na "Cancel" nebo zavřel dialog
                break
            if os.path.exists(directory):
                folder_path.append(directory)
        if len(folder_path) > 0:
            break
        else:
            print("\n\tNebyly vybrány žádné složky.")
    """if folder_path:
        print("Vybraná složka:", folder_path)"""
    return folder_path


def program_shutdown(message: Exception | str = "\n\nUkončení programu", try_save=True):
    if try_save:
        global saved_data_name
        if 'saved_data_name' not in globals():
            saved_data_name = saved_data
        if try_save_data(zip_name=saved_data_name + "_backup") is SaveError:
            print("\n\033[33;1;21mWARRNING\033[0m\n\tZáloha dat se nepovedla.")
    plt.close('all')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print(f"\n\033[1;31;9;21m{'‗' * 55}\033[0m\n{str(message)}")
    sys.exit()


def send_out_message(send_sms=False, send_mail=False, sms_body="[Program is finished]",
                     mail_body="[Program is finished]",
                     name_of_sender="", receiver_phone_number=r"737192215",
                     receiver_email_address=r'matejporubsky@gmail.com', time_stamp=None):
    if not send_sms and not send_mail:
        print("\nNo message was selected and send.")
        return
    else:
        from message_sender import send_message
        print()
        send_message(send_sms=send_sms, send_mail=send_mail, message_sms=sms_body, message_mail=mail_body,
                     name=name_of_sender, phone_number=receiver_phone_number, mail_address=receiver_email_address,
                     formatted_time=time_stamp)

        print("Messaging is done.")


def get_current_date(date_format="%H-%M-%S_%d-%m-%Y"):
    # Získání aktuálního času jako float (sekundy)
    # current_time_float = time.time()

    # Převod aktuálního času na strukturu struct_time
    current_time_struct = time.localtime(time.time())

    # Formátování aktuálního času do požadovaného formátu
    formatted_time = time.strftime(date_format, current_time_struct)
    return formatted_time


def mark_points_on_canvas(index, window_name="Přidání bodů", graph_title="Mark points on image", image=None):
    if image is None:
        img = load_photo(img_index=0, color_type=1)
    else:
        img = image.copy()

    marked_points = []
    h_, w_ = np.int8(0.0017 * height), np.int8(0.0017 * width)

    # Vytvoření figure a osy v matplotlib
    figure, axes = plt.subplots(figsize=(w_, h_), num=str(window_name))
    plt.title(graph_title, wrap=True)

    text = "Označte body pomocí: 'L_MOUSE+CTRL'\nDokončení: 'ESC'\nUkončení programu: 'r'"
    axes.text(0, 0, text, transform=axes.transAxes, fontsize=5,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5))

    # Funkce pro označení bodů
    def on_click(event):
        if event.button == 1 and event.key == 'control':
            x = event.xdata
            y = event.ydata
            if index == 3 or index == 4:
                axes.axvline(x=x, color='royalblue', linestyle='-.')
                axes.axhline(y=y, color='royalblue', linestyle='-.')
            axes.plot(x, y, 'ro')
            figure.canvas.draw()
            if index == 2 and len(marked_points) > 3:
                plt.close()
            else:
                marked_points.append((x, y))

    def on_key(event):
        if event.key == 'escape':
            plt.close()

        elif event.key == 'r':
            program_shutdown(try_save=False)

    # Připojení události ke funkci mark_points
    figure.canvas.mpl_connect('key_press_event', on_key)
    figure.canvas.mpl_connect('button_press_event', on_click)

    # Zobrazení obrázku v matplotlib
    if index == 1:
        # Vytvoření prázdných mask pro obě oblasti
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Vykreslení mnohoúhelníků na maskách
        [cv2.fillPoly(mask, [np.int32(np.round(p))], 255) for p in points_pos]

        # Aplikace mask na obraz
        masked_image1 = img & mask

        # Vytvoření obrazu s intenzitou 0.2
        image_half_intensity = (img * 0.2).astype(np.uint8)

        # Spojení původního obrazu s intenzitou 0.5 a sledovanými oblastmi
        combined_image = cv2.addWeighted(image_half_intensity, 1.0, masked_image1, 0.5, 0)
        axes.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))

    elif index == 4:
        axes.imshow(cv2.cvtColor(load_photo(img_index=-1, color_type=1), cv2.COLOR_BGR2RGB))

    # Zobrazení první
    else:
        axes.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Zobrazení obrázku s označenými body
    plt.tight_layout()
    axes.autoscale(True)
    plt.show()

    if 2 < index <= 4:
        try:
            sorted_points = reorder(marked_points)

            rectangle_points = np.zeros((4, 2), np.int32)
            rectangle_points[0, 0] = rectangle_points[3, 0] = np.round((sorted_points[0, 0] + sorted_points[3, 0]) / 2)
            rectangle_points[0, 1] = rectangle_points[1, 1] = np.round((sorted_points[0, 1] + sorted_points[1, 1]) / 2)
            rectangle_points[1, 0] = rectangle_points[2, 0] = np.round((sorted_points[1, 0] + sorted_points[2, 0]) / 2)
            rectangle_points[2, 1] = rectangle_points[3, 1] = np.round((sorted_points[2, 1] + sorted_points[3, 1]) / 2)
            return rectangle_points

        except (ValueError, IndexError) as e:
            print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba v označení bodů.\n\t➤ POPIS: {e}\n\nZadejte znovu")
            return []
    elif index == 2:
        marked_points = sorted(marked_points, key=lambda point: point[0])

    return marked_points


def edit_points_on_canvas(points_coordinates: list | tuple | np.ndarray, image=None, img_index=0, img_color=0,
                          path_color='red'):
    if image is None:
        img = load_photo(img_index=img_index, color_type=img_color)
    else:
        img = image.copy()
    max_h, max_w = img.shape[:2]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col = None
    else:
        col = 'gray'

    def update_graph(event=None):
        if event is None:
            fig.canvas.draw()
            return
        try:
            index = [int(float(event.widget.winfo_name().split("!entry")[1])) - 1]
        except ValueError:
            index = [0, -1]
        try:
            new_coords = entries[index[0]].get().replace(",", ".").strip('()').strip('[]').split(';')
            if len(new_coords) == 2:
                new_coords = (min(max(float(new_coords[0]), 0), max_w), min(max(float(new_coords[1]), 0), max_h))
                entries[index[0]].delete(0, "end")
                entries[index[0]].insert(
                    0, f'{int(new_coords[0]) if float(new_coords[0]).is_integer() else new_coords[0]} ; '
                       f'{int(new_coords[1]) if float(new_coords[1]).is_integer() else new_coords[1]}')
                points = polygon.get_path().vertices
                points[index] = [new_coords[0], new_coords[1]]
                polygon.set_xy(points)
                texts[index[0]].set_position((new_coords[0] + t_s, new_coords[1] - t_s))
                fig.canvas.draw()
        except (ValueError, NameError):
            print("\n\033[33mWARRNING:\tChyba v získání hodnot vrcholů.\033[0m")
            return

    # Vytvoření hlavního okna Tkinter
    root = tk.Tk()
    root.title("Point coordinates adjustment")

    # Vytvoření rámce pro vstupní textová pole
    input_frame = ttk.Frame(root)
    input_frame.grid(row=0, column=0, padx=10, pady=10)

    entries = []
    for i in range(len(points_coordinates)):
        label = ttk.Label(input_frame, text=f"Bod {i + 1}:")
        label.grid(row=i, column=0, padx=5, pady=5)
        entry = Entry(input_frame)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entry.bind("<Return>", update_graph)
        entry.bind("<FocusOut>", update_graph)
        entries.append(entry)

    # Vytvoření tlačítka pro aktualizaci grafu
    update_button = ttk.Button(input_frame, text="Aktualizovat Graf", command=update_graph)
    update_button.grid(row=i + 1, column=0, columnspan=2, padx=5, pady=10)

    # Vytvoření grafu vpravo
    graph_frame = ttk.Frame(root)
    graph_frame.grid(row=0, column=1, padx=10, pady=10)

    fig, ax = plt.subplots()

    ax.imshow(img, cmap=col)

    polygon = Polygon(points_coordinates, closed=True, fill=False, edgecolor=path_color)
    ax.add_patch(polygon)

    # Vytvoření pole pro TextBoxy
    texts = []

    t_s = max(min(max_h, max_w) * 0.025, 0.1)

    # Vytvoření TextBoxu pro každý vrchol polygonu
    for i, (x, y) in enumerate(points_coordinates):
        entries[i].insert(0, f'{int(x) if float(x).is_integer() else x} ; {int(y) if float(y).is_integer() else y}')
        texts.append(ax.text(x + t_s, y - t_s, f"{i + 1}.", fontweight='bold', color='black', ha='left', va='top',
                             fontsize=7.5, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', pad=0.3,
                                                     alpha=0.45)))

    # ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout(pad=0)

    # Vytvoření navigačního panelu Matplotlib
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Vytvoření navigačního panelu
    toolbar = NavigationToolbar2Tk(canvas, graph_frame)
    toolbar.update()
    canvas_widget.pack()

    # Spuštění hlavní smyčky Tkinter
    root.mainloop()
    plt.close(fig)

    return np.int32(np.round(polygon.get_path().vertices[:-1]))


def edit_object_on_canvas(center_cor: list | tuple | np.ndarray, polygons_cor: list | tuple | np.ndarray,
                          image=None, img_index=0, img_color=0):
    global cur_p_, rotations_, scales_, polygons_, centers_, texts_, entries_, pp_, cc_
    if image is None:
        img = load_photo(img_index=img_index, color_type=img_color)
    else:
        img = image.copy()
    max_h, max_w = img.shape[:2]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col = None
    else:
        col = 'gray'

    def update_graph(_=None):
        submit_x(0)
        submit_y(0)
        submit_s(0)
        submit_sx(0)
        submit_sy(0)
        submit_r(0)

    def submit_point(_):
        global cur_p_, rotations_, scales_, polygons_, centers_, entries_
        try:
            cur_p_ = min(max(int(round(float(entries_[0].get().replace(",", ".")))), 1), tot_areas)
            entries_[0].delete(0, "end")
            entries_[0].insert(0, f"{cur_p_}")
            cur_p_ -= 1
            [(c.set_color('navy'), p.set_color('dodgerblue')) for c, p in zip(centers_, polygons_)]
            polygons_[cur_p_].set_color('orange')
            centers_[cur_p_].set_color('orangered')
            entries_[1].delete(0, "end")
            entries_[1].insert(0, f"{int(centers_[cur_p_].get_offsets()[0][0])}" if float(
                centers_[cur_p_].get_offsets()[0][0]).is_integer() else f"{centers_[cur_p_].get_offsets()[0][0]:.4f}")
            entries_[2].delete(0, "end")
            entries_[2].insert(0, f"{int(centers_[cur_p_].get_offsets()[0][1])}" if float(
                centers_[cur_p_].get_offsets()[0][1]).is_integer() else f"{centers_[cur_p_].get_offsets()[0][1]:.4f}")
            entries_[3].delete(0, "end")
            entries_[3].insert(0, f"{int(rotations_[cur_p_])}" if float(
                rotations_[cur_p_]).is_integer() else f"{rotations_[cur_p_]}")
            entries_[4].delete(0, "end")
            entries_[4].insert(0, f"{int(scales_[cur_p_][0])}" if float(
                scales_[cur_p_][0]).is_integer() else f"{scales_[cur_p_][0]}")
            entries_[5].delete(0, "end")
            entries_[5].insert(0, f"{int(scales_[cur_p_][1])}" if float(
                scales_[cur_p_][1]).is_integer() else f"{scales_[cur_p_][1]}")
            entries_[6].delete(0, "end")
            entries_[6].insert(0, f"{int(scales_[cur_p_][2])}" if float(
                scales_[cur_p_][2]).is_integer() else f"{scales_[cur_p_][2]}")
            fig.canvas.draw()
        except (ValueError, SyntaxError):
            pass

    def submit_x(_):
        global cur_p_, polygons_, centers_, entries_, texts_, pp_, cc_
        try:
            s = min(max(float(entries_[1].get().replace(",", ".")), 0), max_w)
            try:
                s = int(s) if s.is_integer() else s
            except AttributeError:
                pass
            entries_[1].delete(0, "end")
            entries_[1].insert(0, f"{s}")
            s -= cc_[cur_p_][0]

            cc_[cur_p_][0] = cc_[cur_p_][0] + s
            points = polygons_[cur_p_].get_path().vertices
            points[:, 0] = points[:, 0] + s
            polygons_[cur_p_].set_xy(points - (np.mean(points, axis=0) - cc_[cur_p_]))  # korekce
            centers_[cur_p_].set_offsets(cc_[cur_p_])
            texts_[cur_p_].set_position((cc_[cur_p_][0] + t_s, cc_[cur_p_][1] - t_s))
            fig.canvas.draw()
            pp_[cur_p_][:, 0] = pp_[cur_p_][:, 0] + s
        except (ValueError, SyntaxError):
            pass

    def submit_y(_):
        global cur_p_, polygons_, centers_, entries_, texts_, pp_, cc_
        try:
            s = min(max(float(entries_[2].get().replace(",", ".")), 0), max_h)
            try:
                s = int(s) if s.is_integer() else s
            except AttributeError:
                pass
            entries_[2].delete(0, "end")
            entries_[2].insert(0, f"{s}")
            s -= cc_[cur_p_][1]

            cc_[cur_p_][1] = cc_[cur_p_][1] + s
            points = polygons_[cur_p_].get_path().vertices
            points[:, 1] = points[:, 1] + s
            polygons_[cur_p_].set_xy(points - (np.mean(points, axis=0) - cc_[cur_p_]))
            centers_[cur_p_].set_offsets(cc_[cur_p_])
            texts_[cur_p_].set_position((cc_[cur_p_][0] + t_s, cc_[cur_p_][1] - t_s))
            fig.canvas.draw()
            pp_[cur_p_][:, 1] = pp_[cur_p_][:, 1] + s
        except (ValueError, SyntaxError):
            pass

    def submit_r(_):
        global cur_p_, rotations_, scales_, polygons_, centers_, entries_, pp_, cc_
        try:
            rot = round(float(eval(entries_[3].get().replace(",", "."), {'np': np})), 5)
            try:
                rot = int(rot) if rot.is_integer() else rot
            except AttributeError:
                pass
            entries_[3].delete(0, "end")
            if not 0 <= rot < 359:
                entries_[3].insert(0, f"{rot % 360}")
            else:
                entries_[3].insert(0, f"{rot}")
            rotations_[cur_p_] = rot
            rot = np.deg2rad(rot)

            # Vytvoření transformační matice pro rotaci
            rotation_matrix = np.float64([[np.cos(rot), -np.sin(rot)],
                                          [np.sin(rot), np.cos(rot)]]) * scales_[cur_p_][0]
            points = pp_[cur_p_].copy()
            points[:, 0] = cc_[cur_p_][0] + scales_[cur_p_][1] * (pp_[cur_p_][:, 0] - cc_[cur_p_][0])
            points[:, 1] = cc_[cur_p_][1] + scales_[cur_p_][2] * (pp_[cur_p_][:, 1] - cc_[cur_p_][1])
            rot_points = np.dot(points - cc_[cur_p_], rotation_matrix.T) + cc_[cur_p_]
            polygons_[cur_p_].set_xy(rot_points - (np.mean(rot_points, axis=0) - cc_[cur_p_]))
            centers_[cur_p_].set_offsets(cc_[cur_p_])
            fig.canvas.draw()
        except (ValueError, SyntaxError):
            pass

    def submit_s(_):
        global cur_p_, scales_, entries_
        try:
            sc = round(max(float(entries_[4].get().replace(",", ".")), 0.001), 4)
            scales_[cur_p_][0] = sc
            try:
                sc = int(sc) if sc.is_integer() else sc
            except AttributeError:
                pass
            entries_[4].delete(0, "end")
            entries_[4].insert(0, f"{sc}")
            submit_r(0)
        except (ValueError, SyntaxError):
            pass

    def submit_sx(_):
        global cur_p_, scales_, entries_
        try:
            sc = round(max(float(entries_[5].get().replace(",", ".")), 0.001), 4)
            scales_[cur_p_][1] = sc
            try:
                sc = int(sc) if sc.is_integer() else sc
            except AttributeError:
                pass
            entries_[5].delete(0, "end")
            entries_[5].insert(0, f"{sc}")
            submit_r(0)
        except (ValueError, SyntaxError):
            pass

    def submit_sy(_):
        global cur_p_, scales_, entries_
        try:
            sc = round(max(float(entries_[6].get().replace(",", ".")), 0.001), 4)
            scales_[cur_p_][2] = sc
            try:
                sc = int(sc) if sc.is_integer() else sc
            except AttributeError:
                pass
            entries_[6].delete(0, "end")
            entries_[6].insert(0, f"{sc}")
            submit_r(0)
        except (ValueError, SyntaxError):
            pass

    # Vytvoření hlavního okna Tkinter
    root = tk.Tk()
    root.title("Point coordinates adjustment")

    # Vytvoření rámce pro vstupní textová pole
    input_frame = ttk.Frame(root)
    input_frame.grid(row=0, column=0, padx=10, pady=10)

    entries_ = []
    i = 0
    for t, i, f in zip(("Bod", "X", "Y", "Rotace", "Měřítko", "Měřítko x", "Měřítko y"), range(7),
                       (submit_point, submit_x, submit_y, submit_r, submit_s, submit_sx, submit_sy)):
        label = ttk.Label(input_frame, text=f"{t}:")
        label.grid(row=i, column=0, padx=5, pady=5)
        entry = Entry(input_frame)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entry.bind("<Return>", f)
        entry.bind("<FocusOut>", f)
        entries_.append(entry)

    # Vytvoření tlačítka pro aktualizaci grafu
    update_button = ttk.Button(input_frame, text="Aktualizovat Graf", command=update_graph)
    update_button.grid(row=i + 1, column=0, columnspan=2, padx=5, pady=10)

    # Vytvoření grafu vpravo
    graph_frame = ttk.Frame(root)
    graph_frame.grid(row=0, column=1, padx=10, pady=10)

    fig, ax = plt.subplots()

    fig.subplots_adjust(right=0.7, left=0.02, top=0.95, bottom=0.02, wspace=0, hspace=0)

    cc_ = []
    pp_ = []

    polygons_ = []
    centers_ = []
    texts_ = []
    ax.imshow(img, cmap=col)

    t_s = max(min(max_h, max_w) * 0.025, 0.1)

    for i, (center, polygon) in enumerate(zip(center_cor, polygons_cor)):
        polygons_.append(Polygon(polygon, closed=True, fill=True, alpha=0.35))
        centers_.append(ax.scatter(*center, zorder=3, alpha=0.5))
        cc_.append(np.float64(center))
        pp_.append(np.float64(polygon))
        texts_.append(ax.text(center[0] + t_s, center[1] - t_s, f"{i + 1}.", fontweight='bold', color='black',
                              ha='left', va='top', fontsize=5,
                              bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', pad=0.3, alpha=0.45)))
    [ax.add_patch(polygon) for polygon in polygons_]

    tot_areas = len(centers_)

    rotations_ = [0] * tot_areas
    scales_ = np.ones((tot_areas, 3), dtype=float)
    cur_p_ = 0

    [(c.set_color('navy'), p.set_color('dodgerblue')) for c, p in zip(centers_, polygons_)]
    polygons_[cur_p_].set_color('orange')
    centers_[cur_p_].set_color('orangered')

    entries_[0].insert(0, f"{cur_p_ + 1}")
    entries_[1].insert(0, f"{int(centers_[cur_p_].get_offsets()[0][0])}" if float(
        centers_[cur_p_].get_offsets()[0][0]).is_integer() else f"{centers_[cur_p_].get_offsets()[0][0]:.4f}")
    entries_[2].insert(0, f"{int(centers_[cur_p_].get_offsets()[0][1])}" if float(
        centers_[cur_p_].get_offsets()[0][1]).is_integer() else f"{centers_[cur_p_].get_offsets()[0][1]:.4f}")
    entries_[3].insert(0, "0")
    entries_[4].insert(0, "1")
    entries_[5].insert(0, "1")
    entries_[6].insert(0, "1")

    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout(pad=0)

    # Vytvoření navigačního panelu Matplotlib
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Vytvoření navigačního panelu
    toolbar = NavigationToolbar2Tk(canvas, graph_frame)
    toolbar.update()
    canvas_widget.pack()

    # Spuštění hlavní smyčky Tkinter
    root.mainloop()
    plt.close(fig)

    return [c.get_offsets()[0] for c in centers_], [p.get_path().vertices[:-1] for p in polygons_]


def mark_rectangle_on_canvas(window_name="Adding points", graph_title="Mark points on image", edge_color="darkgreen",
                             back_color="yellowgreen", thickness=1.5, shown_photo=0, image=None,
                             cur_num=None, tot_num=None):
    """Mark rectangle on canvas"""
    """if 'matplotlib.widgets.RectangleSelector' not in sys.modules:
        from matplotlib.widgets import RectangleSelector"""

    if tot_num is not None and cur_num is not None:
        graph_title = graph_title + f"  [{cur_num} / {tot_num}]"
    elif tot_num is not None and cur_num is None:
        graph_title = graph_title + f"  [{cur_num}]"

    if image is None:
        img = load_photo(img_index=shown_photo, color_type=1)
    else:
        img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def onselect(_, __):
        pass

    figure, axis = plt.subplots(num=str(window_name))
    plt.title(graph_title, wrap=True)

    axis.imshow(img)
    style = dict(facecolor=back_color, edgecolor=edge_color, alpha=0.25, linestyle='dashed', linewidth=thickness)
    selector = RectangleSelector(axis, onselect, props=style, useblit=True, button=[1],
                                 minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    axis.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    axis.autoscale(True)
    plt.show()

    rectangle = np.float64(selector.extents).reshape(2, 2).T

    # Procházení sloupců a nastavení horní hodnoty
    for column, limit in enumerate((img.shape[1], img.shape[0])):
        rectangle[:, column] = np.clip(rectangle[:, column], a_min=0, a_max=limit)

    return rectangle


def mark_polygon_on_canvas(window_name="Adding points", graph_title="Mark points on image", edge_color="dodgerblue",
                           show_box=False, edge_box_color="darkslategray", back_box_color="skyblue", shown_photo=0,
                           image=None):
    """Mark polygon on canvas"""
    """if 'matplotlib.widgets.PolygonSelector' not in sys.modules:
        from matplotlib.widgets import PolygonSelector"""

    if image is None:
        img = load_photo(img_index=shown_photo, color_type=1)
    else:
        img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def onselect(_):
        pass

    figure, axis = plt.subplots(num=str(window_name))
    plt.title(graph_title, wrap=True)

    axis.imshow(img)

    style = dict(color=edge_color, alpha=0.7, linestyle='dashed', linewidth=1.5)
    if show_box:
        box_style = dict(facecolor=back_box_color, edgecolor=edge_box_color, alpha=0.2, linewidth=1)
    else:
        box_style = None
    selector = PolygonSelector(axis, onselect, props=style, useblit=True, draw_bounding_box=show_box,
                               box_props=box_style)

    axis.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    axis.autoscale(True)
    """plt.cla()
    axis.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(graph_title)"""
    plt.show()

    polygon = np.float64(selector.verts)

    # Procházení sloupců a nastavení horní hodnoty
    if len(polygon) > 0:
        for column, limit in enumerate((img.shape[1], img.shape[0])):
            polygon[:, column] = np.clip(polygon[:, column], a_min=0, a_max=limit)

    return polygon


def mark_ellipse_on_canvas(window_name="Adding points", graph_title="Mark points on image", edge_color="purple",
                           back_color="palevioletred", thickness=1.5, shown_photo=0, image=None,
                           cur_num=None, tot_num=None):
    """Mark Ellipse on canvas"""
    if tot_num is not None and cur_num is not None:
        graph_title = graph_title + f"  [{cur_num} / {tot_num}]" + "\nSet points: 'R_mouse' ; Remove points: 'c'"
    elif cur_num is None:
        graph_title = graph_title + f"  [{cur_num}]" + "\nSet points: 'R_mouse' ; Remove points: 'c'"
    else:
        graph_title = graph_title + "\nSet points: 'R_mouse' ; Remove points: 'c'"

    if image is None:
        img = load_photo(img_index=shown_photo, color_type=1)
    else:
        img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def select_callback(_, __):  # e_click, e_release
        pass

    # Inicializace prázdných seznamů pro ukládání bodů
    body_x, body_y, center_x, center_y = [], [], [], []

    # Funkce pro vykreslení bodu na grafu
    def draw_point(event):
        if event.button == 3:  # Levé tlačítko myši
            plt.scatter(event.xdata, event.ydata, s=30, c='b', zorder=3)  # Vykreslení bodu
            body_x.append(event.xdata)
            body_y.append(event.ydata)

            # Vykreslení úseček spojujících body
            if len(body_x) > 0:
                plt.plot(body_x[-2:], body_y[-2:], c='b', zorder=2)

            # Výpočet a vykreslení středu úsečky
            if len(body_x) > 1:
                center_x.append((body_x[-1] + body_x[-2]) / 2)
                center_y.append((body_y[-1] + body_y[-2]) / 2)
                plt.scatter(center_x[-1], center_y[-1], s=40, facecolor='r', edgecolors='b', zorder=3)
            plt.draw()

    # Funkce pro smazání bodů
    def remove_points(event):
        if event.key == 'c':
            del body_x[:], body_y[:], center_x[:], center_y[:]
            plt.cla()
            axis.imshow(img)
            plt.title(graph_title, wrap=True)
            plt.draw()

    figure, axis = plt.subplots(num=str(window_name))
    plt.title(graph_title, wrap=True)

    axis.imshow(img)

    style = dict(facecolor=back_color, edgecolor=edge_color, alpha=0.45, linestyle='dashed', linewidth=thickness)

    selector = EllipseSelector(axis, select_callback, props=style, useblit=True, button=[1], minspanx=5, minspany=5,
                               spancoords='pixels', interactive=True)

    # Připojení událostí pro klikání myší a stisk klávesy
    figure.canvas.mpl_connect('button_press_event', draw_point)
    figure.canvas.mpl_connect('key_press_event', remove_points)

    axis.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    axis.autoscale(True)
    plt.show()

    center = np.float64(selector.center)
    edge = np.float64((selector.geometry[1], selector.geometry[0])).T

    # Procházení sloupců a nastavení horní hodnoty
    for column, limit in enumerate((img.shape[1], img.shape[0])):
        center[column] = np.clip(center[column], a_min=0, a_max=limit)
        edge[:, column] = np.clip(edge[:, column], a_min=0, a_max=limit)

    return center, edge, selector.extents


def reorder(points):
    if len(points) < 3:
        return points

    points = np.float64(points)
    # Vypočítání středu bodů
    center = np.mean(points, axis=0)

    # Výpočet úhlů bodů od středu
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # Seřazení bodů podle úhlů
    sorted_indices = np.argsort(angles)

    # Přeuspořádání bodů
    sorted_points = points[sorted_indices]

    # Přesunutí prvního bodu na začátek
    first_index = np.where(sorted_indices == 0)[0][0]
    sorted_points = np.roll(sorted_points, -first_index, axis=0).reshape((-1, 2))

    return sorted_points


def pixel_correlation(points, info_text='', divide_area=True, axis_lock=None, x_division=7, y_division=7,
                      min_length_x=50, min_length_y=50):
    print(f"\n\tSpuštění korelace: {info_text}")

    start_time = time.time()

    x_min, y_min = points[0]
    x_max, y_max = points[1]

    x_f_mean, y_f_mean, x_o_mean, y_o_mean = np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,))
    thresholds = []

    if divide_area:
        x = np.linspace(x_min, x_max, max(min(((x_max - x_min) // min_length_x) + 1, x_division), 2), dtype=np.int32)
        y = np.linspace(y_min, y_max, max(min(((y_max - y_min) // min_length_y) + 1, y_division), 2), dtype=np.int32)

        total_correlation = np.int16(np.round((len(x) - 1) * (len(y) - 1)))
    else:
        x, y = [x_min, x_max], [y_min, y_max]
        total_correlation = 1

    current_correlation = 1

    show_areas = False
    if show_areas:  # #########################################################
        gr = gray2.copy()

    for i_x in range(len(x) - 1):
        x1, x2 = x[i_x], x[i_x + 1]
        for i_y in range(len(y) - 1):
            y1, y2 = y[i_y], y[i_y + 1]

            template = gray1[y1:y2, x1:x2]

            # Porovnejte šablonu s druhou fotografií pomocí metody šablony
            result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)

            if str(axis_lock).lower() == 'x' or axis_lock == 0:
                _, max_val, _, max_loc = cv2.minMaxLoc(result[:, i_x])
                max_loc[1] += i_x
            elif str(axis_lock).lower() == 'y' or axis_lock == 1:
                _, max_val, _, max_loc = cv2.minMaxLoc(result[i_y, :])
                max_loc[0] += i_y
            else:
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val < 0.75:
                print(f"\n\t\033[33;1;21mWARRNING\033[0m\n\t\tOblast nenalezena: korelační koeficien  {max_val:.4f}.")
                max_val = 0
                continue

            x_f, y_f = max_loc

            x_f_mean, y_f_mean = np.append(x_f_mean, x_f), np.append(y_f_mean, y_f)
            x_o_mean, y_o_mean = np.append(x_o_mean, x1), np.append(y_o_mean, y1)
            thresholds.append(np.int8(np.round(max_val * 100)))

            print_progress_bar(current_correlation, total_correlation, 1, 20, "\t")

            current_correlation += 1

            if show_areas:  # ########################################################################
                cv2.namedWindow('Original area', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Original area', x2 - x1, y2 - y1)
                cv2.imshow('Original area', template)

                cv2.line(gr, (x1, y1), (x1, y_f + y2 - y1), (255, 0, 0), 5)
                cv2.line(gr, (x2, y1), (x2, y_f + y2 - y1), (255, 0, 0), 5)
                cv2.rectangle(gr, (x1, y1), (x2, y2), (255, 255, 255), 7)
                cv2.rectangle(gr, (x_f, y_f), (x_f + x2 - x1, y_f + y2 - y1), (0, 255, 0), 5)
                cv2.circle(gr, (x_f, y_f), 5, (0, 0, 255), 15)

                cv2.namedWindow('Found area', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Found area', x2 - x1, y2 - y1)
                cv2.imshow('Found area', gr[y_f:y_f + y2 - y1, x_f:x_f + x2 - x1])

                cv2.namedWindow('Found image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Found image', np.int8(0.2 * width), np.int8(0.2 * height))

                print("\n\t\tPress Enter or Space to close the windows.")

                # Zobrazení výsledku
                cv2.imshow('Found image', gr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    x_f_mean, y_f_mean, x_o_mean, y_o_mean = np.mean(x_f_mean), np.mean(y_f_mean), np.mean(x_o_mean), np.mean(y_o_mean)
    x_shifted, y_shifted = x_min + (x_f_mean - x_o_mean), y_min + (y_f_mean - y_o_mean)

    # upper_area = np.int32(
    #    np.round([[x_shifted, y_shifted], [x_shifted + (x_max - x_min), y_shifted + (y_max - y_min)]]))
    upper_area = np.array([[x_shifted, y_shifted], [x_shifted + (x_max - x_min), y_shifted + (y_max - y_min)]])

    print(f"\n\tNalezené oblasti:\n\t\tPřesnost: {np.int8(thresholds)} %"
          f"\n\t\t\tDoba vytváření: {time.time() - start_time: .2f} s.")
    return upper_area


def set_roi(finish_marking=False, just_load=False):
    """Mark ROI"""

    """if 'json' in sys.modules:
        import json"""

    # global correlation_areas, positive_roi, negative_roi, maximal_area, tracking_points
    global points_cor, points_pos, points_neg, points_max, points_track, photo_size

    print("\nOznačení zajmových oblastí:")

    names = ['Points of positive area', 'Points of negative area', 'Points of correlation area',
             'Points for cropping image', 'Points for individual tracking']

    titles = ['Označte alespoň 3 body positivní oblasti.',
              'Označte alespoň 3 body negativní oblasti.',
              'Označte oblast korelace pohyblivé zatěžovací části.',
              'Označte oblast sledované oblasti korelace.',
              'Označte oblast minimální plochy obrazce.',
              'Označte bod sledování']

    done_changes, roi_areas = False, None

    if not finish_marking:
        points_pos = []
        (photo_size, points_neg, points_cor, points_max, points_track) = 5 * [None]
    else:
        just_load = False
    """roi_areas = dict(points_cor=None, positive_roi=None, points_neg=None, points_max=None, 
                     points_track=None)
    for name in roi_areas.keys():
        locals()[name] = roi_areas[name]"""

    img = load_photo(img_index=0, color_type=1)
    masked_img = img.copy()

    if (load_set_points and load_calculated_data and saved_file_exist) and not finish_marking:
        zip_file_name = os.path.join(current_folder_path, saved_data_name + ".zip")
        with zipfile.ZipFile(zip_file_name, 'r') as zipf:
            if not zipf.namelist():  # Zjištění, zda je zip soubor prázdný
                raise MyException(f"\033[31;1;21mError:\033[0m Zip file [{zip_file_name}] is empty.")
            elif 'areas.json' not in zipf.namelist():
                raise MyException(f"\033[31;1;21mError:\033[0m V uložených datech se nenachází soubor oblastí JSON.")
            try:
                # Načítání nastavení z JSON souboru
                print(f'\nPokus o načtení oblastí z uložených dat.')
                with zipf.open('areas.json') as file:
                    roi_areas = json.load(file)
                file.close()
            except (Exception, MyException) as ke:
                print(f'\n\033[33;1;21mWARRNING\033[0m\n\tChyba načtení oblastí JSON\n\tPOPIS: {ke}')

    saved_areas_path = os.path.join(current_folder_path, 'areas.json')
    if (not isinstance(roi_areas, dict) and load_set_points) and not finish_marking:
        # saved_areas_path = os.path.join(current_folder_path, 'areas.npz')
        if os.path.exists(saved_areas_path):
            # Načítání matic
            # loaded_data = np.load(saved_areas_path)
            try:
                print(f'\nPokus o načtení dat z aktuální složky.')
                with open(saved_areas_path, 'r') as file:
                    roi_areas = json.load(file)
                file.close()
            except Exception as ke:
                print(f'\n\033[33;1;21mWARRNING\033[0m\n\tChyba načtení oblastí JSON\n\tPOPIS: {ke}')

    if isinstance(roi_areas, dict) and not finish_marking or just_load:
        if 0 < len(roi_areas) <= 6:
            try:
                print(f"\t- Jsou načteny {'veškeré' if len(roi_areas) == 6 else 'některé'} uložené oblasti.")

                if any(var for var in roi_areas.keys() if var not in globals()):
                    print("\nTyto proměnné nejsou definovány, zkontrolujte jejich název: "
                          f"\n\tSprávné: ['points_cor', 'points_pos', 'points_neg', 'points_max', 'points_track']"
                          f"\n\tŠpatné:  {[var for var in roi_areas.keys() if var not in globals()]}")
                    program_shutdown("\n\033[31;1;21mERROR\033[0m\n\tChyba jmen načtených oblastí.", False)

                # Přiřazení hodnot proměnným podle jmen
                for name in roi_areas.keys():
                    globals()[name] = roi_areas.get(name, None)
                    """kod = f'{name} = {roi_areas.get(name, None)}'
                    # Spusťte kód v rámci lokálního prostoru
                    exec(kod, globals())"""

                if photo_size != list(img.shape[:2]):
                    program_shutdown("\n\033[31;1;21mERROR\033[0m\n\tNesouhlasí velikost fotografie.", try_save=False)

                (points_neg, points_max) = [np.int32(var) if isinstance(var, list) else None
                                            for var in (points_neg, points_max)]

                if isinstance(points_cor, list):
                    points_cor = [np.int32(var) for var in points_cor]
                if isinstance(points_track, list):
                    points_track = [tuple([np.int32(sub_var) for sub_var in var]) for var in points_track]
                if isinstance(points_pos, list):
                    points_pos = [np.int32(var) for var in points_pos]

                if len(roi_areas) < 6 or any(value is None for value in roi_areas.values()):

                    print("\n\tChcete dooznačit neoznačené oblasti?\n\t\t Oblasti: "
                          f"{[name for name, value in roi_areas.items() if value is None]}")
                    while True:
                        if super_speed:
                            ans_to_mark = "N"
                        else:
                            ans_to_mark = askstring("Dooznačit neoznačené oblasti",
                                                    "Chcete dooznačit neoznačené oblasti?\nZadejte Y / N: ")
                            # ans_to_mark = input("\t\tZadejte Y / N: ")
                        if ans_to_mark == "Y":
                            print("\n\tZvolena možnost 'Y'")
                            break
                        elif ans_to_mark == "N":
                            print("\n\tZvolena možnost 'N'")
                            return divide_image(points_pos, points_neg, size)
                            # break
                        else:
                            print("\n Zadejte platnou odpověď.")
            except Exception as e:
                print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba načtení oblastí.\n\tPOPIS: {e}")
        else:
            # program_shutdown("\n\033[31;1;21mERROR\033[0m\n\tChyba u načtení dat oblastí.", try_save=False)
            print("\n\033[31;1;21mERROR\033[0m\n\tChyba načtenych oblastí.\n\tPOPIS: Nesprávný počet načtených dat.")
    else:
        if load_set_points:
            print(f'\n\t\033[33;1;21mNebyla načtena žádná data definovaných oblastí.\033[0m')

    if just_load:
        return None, None

    if isinstance(points_cor, list) and len(points_cor) > 0:
        for points in points_cor:
            rectangle_region = (masked_img[points[0, 1]:points[1, 1],
                                points[0, 0]:points[1, 0]])
            rectangle_region //= 2
            cv2.rectangle(masked_img, points[0], points[1], (255, 255, 255), 2)
            cv2.rectangle(masked_img, points[0], points[1], 0, 1)
    if (isinstance(points_pos, list) and len(points_pos) > 0 and isinstance(points_pos[0], np.ndarray) and
            len(points_pos[0]) > 2):
        for points in points_pos:
            mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            masked_img = cv2.addWeighted((masked_img * 0.3).astype(np.uint8), 1.0,
                                         cv2.bitwise_and(masked_img, masked_img, mask=mask), 0.5, 0)
    if isinstance(points_neg, np.ndarray) and len(points_neg) > 2:
        mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points_neg], 255)
        masked_img[mask > 0] = (masked_img[mask > 0] * 0.35).astype(np.uint8)
    if isinstance(points_track, list) and len(points_track) > 0:
        # i = 1
        for marked_center, marked_area in points_track:
            mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [marked_area], 255)
            masked_img[mask > 0] = (masked_img[mask > 0] * 0.5).astype(np.uint8)
            mask = cv2.add(masked_img, 110, mask=mask)
            masked_img = cv2.bitwise_or(masked_img, mask)
            cv2.circle(masked_img, (marked_center[0], marked_center[1]), 7, (255, 50, 0), -1)

            """cv2.putText(masked_img, f'{i}', (marked_center[0] + 10, marked_center[1] - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 3, (255, 50, 50), 3, cv2.LINE_AA)
            i += 1

        points_track = edit_object_on_canvas([c[0] for c in points_track], [a[1] for a in points_track],
                                             image=img)
        points_track = [(np.int32(np.round(c)), np.int32(np.round(a))) for c, a in
                        zip(points_track[0], points_track[1])]

        done_changes = True

        plt.figure()
        plt.imshow(masked_img, cmap='gray')
        plt.show()"""

    if ((do_auto_mark and not isinstance(points_cor, list)) or
            (isinstance(points_cor, list) and len(points_cor) == 0 and do_calculations['Do Correlation'])):
        print("\t- Je použito automatické označení korelační oblasti.")
        points_cor = []
        global templates_path
        if dynamic_mode:
            templates_path = browse_directory(window_title="Vyberte složku s fotografiemi k pro automatickou korelaci")

        for template in [os.path.join(templates_path, p) for p in os.listdir(templates_path) if
                         os.path.isfile(os.path.join(str(templates_path), p)) and
                         p.lower().endswith(photos_types)]:
            template = cv2.imread(template, 0)
            top_left1, height1, width1, _ = match(template, gray1, tolerance=0.3)
            if top_left1 is not None:
                points_cor.append(np.int32(np.round(np.array(
                    ((top_left1[0], top_left1[1]), (top_left1[0] + width1, top_left1[1] + height1))))))
                done_changes = True

            for points in points_cor:
                rectangle_region = (masked_img[points[0, 1]:points[1, 1],
                                    points[0, 0]:points[1, 0]])
                rectangle_region //= 2
                cv2.rectangle(masked_img, points[0], points[1], (255, 255, 255), 2)
                cv2.rectangle(masked_img, points[0], points[1], 0, 1)
        # np.savez(saved_areas_path, **{f"var_{i + 1}": matrix for i, matrix in enumerate([points_correlation])})

    if (((mark_points_by_hand or finish_marking) and not isinstance(roi_areas, dict)) or (
            isinstance(roi_areas, dict) and (len(roi_areas) < 5 or any(val is None for val in roi_areas.values())))):
        print("\t- Ruční označení korelační oblasti.\n")

    if ((not isinstance(points_cor, list) and mark_points_by_hand) or
            (isinstance(points_cor, list) and len(points_cor) == 0)):
        while True:
            if super_speed:
                cor_num = 1
            else:
                cor_num = askinteger("Zvolte počet korelačních oblastí",
                                     "Zvolte počet korelačních oblastí.\nZadejte číslo: ")
                # cor_num = input("\t\tZvolte počet korelačních oblastí: ").replace(",", ".")
            try:
                cor_num = np.int8(abs(round(np.float16(cor_num))))  # pokus o převod na číslo
                break
            except ValueError as ve:
                print(f"\n Zadejte platnou odpověď.\n\tPOPIS: {ve}")
                pass

        if cor_num > 0:
            points_cor = []
        done_changes = True

        for i, j in enumerate([2] + (cor_num - 1) * [3]):
            while True:
                marked_points = np.int32(np.round(mark_rectangle_on_canvas(
                    names[2], titles[j], "navy", "deepskyblue", image=masked_img, tot_num=cor_num, cur_num=i + 1)))
                if marked_points.all() != np.array(((0, 0), (0, 1))).all():
                    points_cor.append(marked_points)
                    """rectangle_region = (masked_img[marked_points[0, 1]:marked_points[1, 1],
                                        marked_points[0, 0]:marked_points[1, 0]])
                    # Ztmavení oblasti obdélníku na 50 %
                    rectangle_region //= 2"""
                    masked_img[marked_points[0, 1]:marked_points[1, 1], marked_points[0, 0]:marked_points[1, 0]] //= 2
                    cv2.rectangle(masked_img, marked_points[0], marked_points[1], (255, 255, 255), 2)
                    cv2.rectangle(masked_img, marked_points[0], marked_points[1], 0, 1)
                    break

    if do_just_correlation and done_changes:
        roi_areas = dict(photo_size=img.shape[:2], points_cor=points_cor, points_pos=None, points_neg=None,
                         points_max=None, points_track=None)

        roi_areas['points_cor'] = [item if isinstance(item, list) else item.tolist() for item in
                                   roi_areas['points_cor'] if
                                   roi_areas['points_cor'] is not None]
        # Uložení matic
        with open(os.path.join(current_folder_path, 'areas.json'), 'w') as file:
            json.dump(roi_areas, file)
        file.close()
        return None, None

    if (((not isinstance(points_pos, list) or len(points_pos) == 0) and
         mark_points_by_hand) or (isinstance(points_pos[0], np.ndarray) and len(points_pos[0]) == 0 and
                                  len(points_pos) == 0 and
                                  any((do_calculations['Do Rough detection'], do_calculations['Do Fine detection'],
                                       do_calculations['Do Point detection'])))):

        while True:
            p_num = askinteger("Počet hledaných bodů", "Počet hledaných bod.\nZadejte číslo: ")
            try:
                p_num = np.int16(abs(round(np.float16(p_num))))  # pokus o převod na číslo
                break
            except ValueError as ve:
                print(f"\n Zadejte platnou odpověď.\n\tPOPIS: {ve}")
                pass

        # Spojení původního obrazu s intenzitou 0.5 a sledovanými oblastmi
        masked_img = cv2.addWeighted(masked_img, 0.25, masked_img, 0.25, 0)

        for _ in range(p_num):
            while True:
                pos = np.int32(np.round(mark_polygon_on_canvas(
                    names[0], f"Area: [{_ + 1} / {p_num}]  ; " + titles[0], edge_color="darkgreen", show_box=False,
                    edge_box_color="olive", back_box_color="yellowgreen", image=masked_img)))
                if len(pos) > 2:
                    # Ztmavení fotografie
                    mask = np.zeros_like(img)
                    cv2.fillPoly(mask, [pos], (255, 255, 255))  # Vykreslení mnohoúhelníků na maskách
                    # Spojení původního obrazu s intenzitou 0.5 a sledovanými oblastmi
                    """masked_img = cv2.addWeighted((masked_img * 0.3).astype(np.uint8), 1.0,
                                                 cv2.bitwise_and(masked_img, masked_img, mask=mask), 0.5, 0)"""
                    masked_img = cv2.addWeighted(masked_img, 1, mask, 0.5, 0)
                    break
            pos = edit_points_on_canvas(pos, image=masked_img, path_color="darkgreen")
            points_pos.append(pos)

    if (not isinstance(points_neg, np.ndarray) and all(isinstance(p, np.ndarray) for p in points_pos)
            and mark_points_by_hand):

        points_neg = np.int32(np.round(mark_polygon_on_canvas(
            names[1], titles[1], edge_color="red", show_box=False, edge_box_color="firebrick", back_box_color="peru",
            image=masked_img)))

        if len(points_neg) > 2:
            # Ztmavení fotografie
            mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points_neg], 255)  # Vykreslení mnohoúhelníků na maskách
            # Spojení původního obrazu s intenzitou 0.5 a sledovanými oblastmi

            masked_img[mask > 0] = (masked_img[mask > 0] * 0.35).astype(np.uint8)
            points_neg = edit_points_on_canvas(points_neg, image=masked_img, path_color="red")
        done_changes = True

    if not isinstance(points_max, np.ndarray) and mark_points_by_hand:

        print("\n\tChcete oříznout plochu sledované plochy fotografie?")
        while True:
            ans = askstring("Chcete oříznout plochu sledované plochy fotografie?",
                            "Chcete oříznout plochu sledované plochy fotografie?\nZadejte Y / N: ")
            # ans = input("\t\tZadejte Y / N: ")
            if ans == "Y":
                print("\n\tZvolena možnost 'Y'")

                while True:
                    first_points_max = np.int32(np.round(mark_rectangle_on_canvas(names[3], titles[4], "red",
                                                                                  "firebrick", 2.5, image=img)))
                    if first_points_max.all() != np.array(((0, 0), (0, 1))).all():
                        break

                while True:
                    last_points_max = np.int32(np.round(mark_rectangle_on_canvas(names[3], titles[4], "red",
                                                                                 "firebrick", 2.5, shown_photo=-1)))
                    if last_points_max.all() != np.array(((0, 0), (0, 1))).all():
                        break
                points_max = np.int32([np.minimum(first_points_max[0], last_points_max[0]),
                                       np.maximum(first_points_max[1], last_points_max[1])])
                break

            elif ans == "N":
                print("\n\tZvolena možnost 'N'")
                points_max = []
                break

            else:
                print("\n Zadejte platnou odpověď.")
        done_changes = True

    if ((not isinstance(points_track, list) and mark_points_by_hand) or
            (isinstance(points_track, list) and len(points_track) == 0 and
             do_calculations['Do Point detection'])):

        points_track = []
        masked_img2 = masked_img.copy()
        print(" ")
        while True:
            p_num = askinteger("Počet hledaných bodů", "Počet hledaných bod.\nZadejte číslo: ")
            # p_num = input("\t\tZvolte počet hledaných bodů: ").replace(",", ".")
            try:
                p_num = np.int8(abs(round(np.float16(p_num))))  # pokus o převod na číslo
                break
            except ValueError as ve:
                print(f"\n Zadejte platnou odpověď.\n\tPOPIS: {ve}")
                pass

        # masked_img = cv2.add(img, masked_img)
        masked_img = img.copy()
        mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
        if isinstance(points_cor, list) and len(points_cor) > 0:
            for points in points_cor:
                masked_img[points[0, 1]:points[1, 1], points[0, 0]:points[1, 0]] = (
                        masked_img[points[0, 1]:points[1, 1], points[0, 0]:points[1, 0]] * 0.75).astype(np.uint8)
                cv2.rectangle(masked_img, points[0], points[1], (255, 255, 255), 2)
                cv2.rectangle(masked_img, points[0], points[1], 0, 1)
        if points_pos:  # isinstance(points_pos, np.ndarray) and len(points_pos) > 2:
            [cv2.fillPoly(mask, [p], 255) for p in points_pos if isinstance(p, np.ndarray) and len(p) > 2]
        if isinstance(points_neg, np.ndarray) and len(points_neg) > 2:
            cv2.fillPoly(mask, [points_neg], 0)

        # Ztmavení druhé fotografie na 75%
        img = cv2.addWeighted(masked_img, 0.65, np.zeros_like(masked_img), 0.25, 0)
        # Aplikace masky
        masked_img = cv2.bitwise_and(masked_img, masked_img, mask=mask)
        masked_img += cv2.bitwise_and(img, img, mask=~mask)  # ~mask inverts the mask

        for i in range(p_num):
            # masked_img = (masked_img * 0.8).astype(np.uint8)
            while True:
                marked_center, marked_area, marked_extents = mark_ellipse_on_canvas(names[4], titles[5], cur_num=i + 1,
                                                                                    image=masked_img, tot_num=p_num)
                if not (marked_center == [0.0, 0.0]).all() and not marked_extents == (0.0, 0.0, -0.5, 0.5):
                    points_track.append((marked_center, marked_area))
                    marked_center = np.int32(np.round(marked_center))
                    marked_area = np.int32(np.round(marked_area))
                    # Zbarvení fotografie
                    mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [marked_area], 255)  # Vykreslení polygonu na maskách
                    masked_img[mask > 0] = (masked_img[mask > 0] * 0.5).astype(np.uint8)
                    # masked_img[mask > 0] += np.uint8((220, 70, 150))
                    mask = cv2.add(masked_img, 110, mask=mask)  # Přidání hodnoty do modré
                    masked_img = cv2.bitwise_or(masked_img, mask)
                    cv2.circle(masked_img, (marked_center[0], marked_center[1]), 5, (255, 255, 255), -1)
                    cv2.circle(masked_img, (marked_center[0], marked_center[1]), 3, (255, 100, 50), -1)
                    cv2.putText(masked_img, f'{i + 1}', (marked_center[0] + 4, marked_center[1] - 4),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(masked_img, f'{i + 1}', (marked_center[0] + 4, marked_center[1] - 4),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
                    break

        if p_num > 0:
            points_track = edit_object_on_canvas([c[0] for c in points_track], [a[1] for a in points_track],
                                                 image=masked_img2)
            points_track = [(np.int32(np.round(c)), np.int32(np.round(a))) for c, a in
                            zip(points_track[0], points_track[1])]

        done_changes = True
    """if current_image_folder == "_test_1":
        points_pos = np.int16([[2492, 1113], [1902, 2125], [2492, 3151],
                                 [3666, 3151], [4257, 2137], [3666, 1113]])

        points_neg = np.int16([[2523, 1165], [2272, 1595], [3203, 2142], [3173, 2185], [2248, 1648],
                                 [1965, 2128], [2523, 3097], [3635, 3097], [3883, 2666], [2954, 2125],
                                 [2982, 2081], [3913, 2619], [4192, 2138], [3635, 1165]])

        points_cor = [np.int16([[2600, 400], [3450, 650]])]

        points_max = np.int16([[1400, 1050], [4600, 3200]])

    elif current_image_folder == "_test_2":
        points_pos = np.int16(
            [[2594, 1073], [1981, 2124], [2586, 3180], [3803, 3188], [4422, 2137], [3812, 1075]])

        points_neg = np.int16(
            [[2631, 1135], [2050, 2123], [2618, 3123], [3764, 3134], [4352, 2138], [3778, 1137]])

        points_cor = [np.int16([[2730, 350], [3550, 610]])]

        points_max = np.int16([[1590, 980], [4840, 3270]])
    else:
        program_shutdown("Špatná definice pro označení bodů.", try_save=False)"""

    del img

    if done_changes:
        # Vytvoření matic
        roi_areas = dict(photo_size=masked_img.shape[:2], points_cor=points_cor, points_pos=points_pos,
                         points_neg=points_neg, points_max=points_max, points_track=points_track)
        for name in roi_areas.keys():
            try:
                if roi_areas[name] is None:
                    continue
                elif isinstance(roi_areas[name], list):
                    roi_areas[name] = [
                        item if isinstance(item, list) else
                        [
                            sub_item if isinstance(sub_item, list)
                            else sub_item.tolist() for sub_item in item
                        ]
                        if isinstance(item, tuple) else item.tolist() for item in roi_areas[name]]
                elif isinstance(roi_areas[name], tuple):
                    pass
                else:
                    roi_areas[name] = roi_areas[name].tolist()
            except AttributeError:
                roi_areas[name] = None

        # Uložení matic
        # np.savez(saved_areas_path, **{f"var_{i + 1}": matrix for i, matrix in enumerate(matrices)})
        with open(saved_areas_path, 'w') as file:
            json.dump(roi_areas, file)
        file.close()

    # found_mesh, found_triangle_centers = divide_image(points_pos, points_neg, size)

    return divide_image(points_pos, points_neg, size)


def divide_image(area1, area2=None, mesh_size=300, show_graph=True, printout=True):
    """Divide ROI to triangle elements"""

    """if 'pygmsh' not in sys.modules:
        import pygmsh"""

    n = len(area1)

    if area2 is None:
        area2 = []

    triangle_centers = []
    triangle_points = []
    triangle_indexes = []
    mesh = []

    for p in area1:
        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_max = mesh_size
            geom.characteristic_length_min = mesh_size

            """point_entities = [geom.add_point(point, mesh_size) for point in area1]
            point_entities.append(point_entities[0])
            s1 = geom.add_spline(point_entities)
            l1 = geom.add_curve_loop([s1])
            poly1 = geom.add_plane_surface(l1)"""

            poly1 = geom.add_polygon(p,  # mesh_size=mesh_size
                                     )

            if len(area2) > 2:
                """point_entities = [geom.add_point(point, mesh_size) for point in area2]
                point_entities.append(point_entities[0])
                s2 = geom.add_spline(point_entities)
                l2 = geom.add_curve_loop([s2])
                poly2 = geom.add_plane_surface(l2)"""

                poly2 = geom.add_polygon(
                    area2,  # mesh_size=0.1,
                )

                geom.boolean_difference(poly1, poly2)

            m = geom.generate_mesh(dim=2)

            triangle_centers.append(np.mean(m.points[m.get_cells_type("triangle")][:, :, :2], axis=1))
            triangle_points.append(m.points)
            triangle_indexes.append(m.get_cells_type("triangle"))
            mesh.append(m)

    if printout:
        print("\n\tTriangulation is done.\n")
        print("\tVytvořeno", np.sum([len(c) for c in triangle_centers]), "elementů.")

    try:
        img = gray1
    except NameError:
        img = load_photo(0, 0)
    h, w = img.shape[:2]

    ratio = (img.shape[1] / img.shape[0]) * 1.5
    fig_size = 6

    if show_graph:
        from itertools import cycle

        plt.close("Regions of interest")
        # Obraz elemntů na fotografii
        # Vytvoření figure a osy v matplotlib
        plt.figure(figsize=(fig_size * ratio, fig_size), num="Regions of interest")
        plt.suptitle("Triangled ROI area")

        # Definice umístění pro každý graf
        ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=3, rowspan=2)
        ax2 = plt.subplot2grid((2, 4), (0, 3))
        ax3 = plt.subplot2grid((2, 4), (1, 3))

        ax1.imshow(img, cmap='gray')
        color_cycle = cycle(['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 'tab:pink'])

        for i in range(n):
            ax1.triplot(triangle_points[i][:, 0], triangle_points[i][:, 1], triangle_indexes[i],
                        color=next(color_cycle))
            #                                                mesh.get_cells_type("triangle") = mesh.cells[1].data
            ax1.scatter(triangle_centers[i][:, 0], triangle_centers[i][:, 1], s=5, c='orange', marker='o')

        ax1.set_aspect('equal', adjustable='box')
        ax1.autoscale(True)

        print("\t\tFirst graph: finished.")

        # Obraz barevných elementů
        # Vykreslení všech trojúhelníků
        for i in range(n):
            [ax2.triplot(triangle_points[i][cell, 0], triangle_points[i][cell, 1]) for cell in triangle_indexes[i]]
            """ax2.scatter(triangle_centers[:, 0], triangle_centers[:, 1],
                        s=20,  # Velikost bodů
                        c='orange',  # Barva bodů
                        marker='o'  # Znak bodů
                        )"""
        ax2.invert_yaxis()
        ax2.set_aspect('equal', adjustable='box')
        ax2.autoscale(True)

        print("\t\tSecond graph: finished.")

        triangle = min(10, len(triangle_indexes[0]) - 1)  # který trojúhelník chci vykreslit

        triangle_cor = triangle_points[0][triangle_indexes[0][triangle]]

        # Vytvoření prázdných mask pro obě oblasti
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Vykreslení mnohoúhelníků na maskách
        cv2.fillPoly(mask, [np.int32(np.round(triangle_cor[:, :2]))], 255)

        x_max, x_min = (np.int32(np.round(max(triangle_cor[:, 0]) + 10)),
                        np.int32(np.round(min(triangle_cor[:, 0]) - 10)))
        y_max, y_min = (np.int32(np.round(max(triangle_cor[:, 1]) + 10)),
                        np.int32(np.round(min(triangle_cor[:, 1]) - 10)))
        masked_image = (img & mask)[y_min:y_max, x_min:x_max]

        ax3.imshow(masked_image, cmap='gray')
        # Obraz jednoho maskovaného elementu
        ax3.scatter(triangle_centers[0][triangle, 0] - x_min, triangle_centers[0][triangle, 1] - y_min, s=50,
                    c='red', marker='s')
        ax3.axis('off')
        ax3.set_aspect('equal', adjustable='box')
        ax3.autoscale(True)

        print("\t\tThird graph: finished.\n")

        plt.subplots_adjust(right=0.99, left=0.1, top=0.9, bottom=0.1, wspace=0.2, hspace=0.5)
        # plt.tight_layout()

        plt.pause(0.5)
        plt.show(block=block_graphs)
        plt.pause(2)
    return mesh, triangle_centers


def make_eroded_mask(photo, average_area=25, threshold_value=6, dilate_area=50, style=0):
    if style == 0:
        # Laplacian Edges
        laplacian_kernel = np.float32([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        detection_image = cv2.filter2D(photo.copy(), cv2.CV_8U, laplacian_kernel)  # Detekce hran ostré
    elif style == 1:
        # Gradient Edges
        sobel_x = cv2.Sobel(photo, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(photo, cv2.CV_16S, 0, 1, ksize=3)
        detection_image = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5,
                                          cv2.convertScaleAbs(sobel_y), 0.5, 0)
        average_area = 15
        threshold_value = 50
    else:
        return np.ones_like(photo)

    kernel = np.ones((average_area, average_area), dtype=np.float32) / (average_area ** 2)
    blurred_image = cv2.filter2D(detection_image, -1, kernel)

    _, binary_mask = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Morfologická operace dilatace pro spojení blízkých hran
    kernel = np.ones((dilate_area, dilate_area), np.uint8)
    dilated_mask2 = cv2.dilate(binary_mask, kernel, iterations=3)

    # Morfologická operace eroze pro odstranění malých objektů a zúžení hran
    mask2_eroded_lap = cv2.erode(dilated_mask2, kernel, iterations=2)
    return mask2_eroded_lap


def point_locator(mesh, current_shift=None, shift_start=None, m1=None, m2=None, state=0,
                  n_features=0, n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10, sigma=1.6):
    global auto_crop
    global keypoints1_sift, descriptors1_sift

    sift = cv2.SIFT_create(
        nfeatures=n_features,  # __________________ Počet detekovaných rysů (0 = všechny dostupné) ______ def = 0
        nOctaveLayers=n_octave_layers,  # _________ Počet vrstev v každé oktávě _________________________ def = 3
        contrastThreshold=contrast_threshold,  # __ Práh kontrastu pro platnost rysu ____________________ def = 0.04
        edgeThreshold=edge_threshold,  # __________ Práh hrany pro platnost rysu blízko k okraji ________ def = 10
        sigma=sigma  # ___________________________ Gaussovská hladina oktáv ____________________________ def = 1.6
    )
    x_old, y_old, x_new, y_new, angle_old, angle_new, index = [], [], [], [], [], [], [0]
    counter = 0

    length = len(mesh.get_cells_type("triangle"))

    mask1 = m1 if isinstance(m1, np.ndarray) else None
    mask2 = m2 if isinstance(m2, np.ndarray) else None

    if not isinstance(mask1, np.ndarray):
        mask1 = np.zeros(gray1.shape[:2], dtype=np.uint8)
        [cv2.fillPoly(mask1, [np.int32(np.round(p))], 255) for p in points_pos]
        try:
            cv2.fillPoly(mask1, [points_neg], 0)
        except cv2.error:
            pass

        if state == 0:
            print("\n\tChcete použít automatické oříznutí fotografie sledované plochy?")
            while True:
                if super_speed:
                    ans = "Y"
                else:
                    ans = askstring("Chcete použít automatické oříznutí fotografie sledované plochy?",
                                    "Chcete použít automatické oříznutí fotografie sledované plochy?\nZadejte Y / N: ")
                    # ans = input("\t\tZadejte Y / N: ")
                if ans == "Y":
                    print("\t\tZvolena možnost 'Y'")
                    auto_crop = True
                    break

                elif ans == "N":
                    print("\t\tZvolena možnost 'N'")
                    auto_crop = False
                    break

                else:
                    print("\n Zadejte platnou odpověď.")

    if auto_crop and not isinstance(mask2, np.ndarray):
        mask2 = np.zeros(gray2.shape[:2], dtype=np.uint8)
        if (isinstance(shift_start, (np.ndarray, list, tuple)) and len(shift_start) > 0 and
                isinstance(current_shift,
                           (int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64))):
            if not isinstance(shift_start, tuple):
                shift_start = shift_start.copy()
            shift_start = np.int32(np.round(shift_start))
            cv2.rectangle(mask2, (shift_start[0, 0], np.int32(np.round(shift_start[0, 1] + current_shift))),
                          (shift_start[1, 0], shift_start[1, 1]), 255, -1)
            mask2 = cv2.bitwise_and(make_eroded_mask(photo=gray2), mask2)
        else:
            mask2 = make_eroded_mask(photo=gray2)

        if state == 0:
            print("\n\tChcete zobrazit oříznutou fotografii?")
            while True:
                if super_speed:
                    ans = "N"
                else:
                    ans = askstring("Chcete zobrazit oříznutou fotografii?",
                                    "Chcete zobrazit oříznutou fotografii?\nZadejte Y / N: ")
                    # ans = input("\t\tZadejte Y / N: ")
                if ans == "Y":
                    print("\t\tZvolena možnost 'Y'")
                    try:
                        plt.figure(figsize=(10, 7))
                        plt.imshow(cv2.cvtColor(gray2 & mask2, cv2.COLOR_BGR2RGB))
                        plt.tight_layout()
                        plt.gca().autoscale(True)
                        plt.gca().set_aspect('equal', adjustable='box')
                        plt.pause(0.5)
                        plt.show(block=block_graphs)
                        plt.pause(2)
                    except TypeError as te:
                        print(f"\n\033[31;1;21mERROR\033[0m"
                              f"\n\tChyba vykreslení masky 2 (pravděpodobně je '{None}')\n\t- POPIS: {te}")
                    break

                elif ans == "N":
                    print("\t\tZvolena možnost 'N'")
                    break

                else:
                    print("\n Zadejte platnou odpověď.")

    print("\n\tVytváření hledaných bodů.")
    start_time = time.time()

    if state == 0:
        keypoints1_sift, descriptors1_sift = sift.detectAndCompute(gray1, mask1)
        if np.array_equal(gray1, gray2) and not isinstance(mask2, np.ndarray):
            mask2 = mask1

    keypoints2_sift, descriptors2_sift = sift.detectAndCompute(gray2, mask2)

    print("\tHledané body vytvořeny.\n")

    if False:
        plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.0017 * height)))
        plt.title("Keypoints")
        plt.imshow(cv2.drawKeypoints(gray2, keypoints2_sift, gray2), cmap='gray')
        plt.tight_layout()
        plt.axis('equal')
        plt.show()

    print("\t\tElements: ")
    # Cyklus na detekci trojúhelníků
    for triangle in mesh.get_cells_type("triangle"):

        triangle_points = mesh.points[triangle]

        mask1 = np.zeros(gray1.shape[:2], dtype=np.uint8)

        # Vykreslení mnohoúhelníků na maskách
        cv2.fillPoly(mask1, [np.int32(np.round(triangle_points[:, :2]))], 255)

        # Porovnání popisovačů pomocí algoritmu BFMatcher
        bf = cv2.BFMatcher()

        def select(key, des, mask):
            selected_key = [key[idx] for idx, kp in enumerate(key) if
                            mask[np.int32(np.round(kp.pt[1])), np.int32(np.round(kp.pt[0]))]]
            selected_des = des[[idx for idx, kp in enumerate(key) if
                                mask[np.int32(np.round(kp.pt[1])), np.int32(np.round(kp.pt[0]))]]]
            return selected_key, selected_des

        # Nalezení klíčových bodů a popisovačů pro oba obrazy - POUZE těch v MASCE1
        selected_keypoints, selected_descriptors = select(keypoints1_sift, descriptors1_sift, mask1)
        matches = bf.knnMatch(selected_descriptors, descriptors2_sift, k=2)

        # matches = sorted(matches, key=lambda x: x[0].distance)

        # Filtrování unikátních bodů na první fotografii a ukládání odpovídajících bodů na druhé fotografii
        unique_keypoints1 = []
        corresponding_keypoints2 = []
        for m, n in matches:
            if m.distance < precision * n.distance:  # Pomocí této podmínky filtrování shodných bodů
                # Přidání pouze unikátních bodů (bez duplikátů) na první fotografii
                if selected_keypoints[m.queryIdx].pt not in [kp.pt for kp in unique_keypoints1]:
                    unique_keypoints1.append(selected_keypoints[m.queryIdx])
                    corresponding_keypoints2.append(keypoints2_sift[m.trainIdx])

        """# Vykreslení bodů na první fotografii
        output_image1 = cv2.drawKeypoints(gray1, unique_keypoints1, None)
        output_image2 = cv2.drawKeypoints(gray2, corresponding_keypoints2, None)"""

        """if state == 0:
            coordinates_image1 = np.array([kp.pt for kp in unique_keypoints1])
            coordinates_image2 = np.array([kp.pt for kp in corresponding_keypoints2])

            # Zobrazení výsledných obrázků s unikátními body na první fotografii a druhé fotografii
            plt.figure(figsize=(10, 7))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(gray1, cv2.COLOR_BGR2RGB))
            plt.scatter(coordinates_image1[:, 0], coordinates_image1[:, 1], s=20)
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(gray2, cv2.COLOR_BGR2RGB))
            plt.scatter(coordinates_image2[:, 0], coordinates_image2[:, 1], s=20)
            plt.tight_layout()
            plt.show()"""

        # Aplikace prahu na shody mezi popisovači
        good_matches = [m for m, n in matches if m.distance < precision * n.distance]

        # seřazení podle přesnosti
        good_matches.sort(key=lambda x: x.distance)

        """matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(descriptors1_sift, descriptors2_sift, k=2)"""

        """matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(selected_descriptors, descriptors2_sift)
        good_matches = matches"""

        # TODO opravit unikátní body
        """x = [selected_keypoints[m.queryIdx].pt[0] for m in good_matches]
        y = [selected_keypoints[m.queryIdx].pt[1] for m in good_matches]

        coordinates = np.column_stack((x, y))

        # Zbavení se duplicitních bodů
        indexes = np.unique(coordinates, return_index=True, axis=0)[1]
        unique_matches = [good_matches[index] for index in sorted(indexes)]
        unique_matches = unique_matches[:limit_of_points]
        coordinates_image1 = np.array([kp.pt for kp in unique_keypoints1])
        coordinates_image2 = np.array([kp.pt for kp in corresponding_keypoints2])"""

        # omezení počtu dat podle limitu: "limit_of_points"
        limit_of_points = min(points_limit, len(unique_keypoints1))

        x_old.extend([kp.pt[0] for kp in unique_keypoints1][:limit_of_points])
        y_old.extend([kp.pt[1] for kp in unique_keypoints1][:limit_of_points])
        x_new.extend([kp.pt[0] for kp in corresponding_keypoints2][:limit_of_points])
        y_new.extend([kp.pt[1] for kp in corresponding_keypoints2][:limit_of_points])
        # angle_old.extend([selected_keypoints[m.queryIdx].angle for m in good_matches])
        # angle_new.extend([keypoints2[m.trainIdx].angle for m in good_matches])
        index.append(index[counter] + limit_of_points)

        """if counter == 0:
            print("\t\tElement", counter + 1, "hotov.\t\t[", counter + 1, "/", length, "]")
        elif counter % 100 == 0:
            print("\t\tElement", counter, "hotov.\t\t[", counter, "/", length, "]")
        elif counter + 1 == length:
            print("\t\tElement", counter + 1, "hotov.\t\t[", counter + 1, "/", length, "]")"""

        if counter % 20 == 0 or counter + 1 == length:
            print_progress_bar(counter + 1, length, 1, 20, "\t\t\t")

        counter += 1

        if False:
            # Vykreslení bodů daného elemenetu
            # Vytvoření seznamu indexů odpovídajících klíčových bodů
            # matching_indices = [m.queryIdx for m in good_matches]

            # Vytvoření seznamu souřadnic odpovídajících good_matches
            # matched_keypoints1 = [selected_keypoints[m.queryIdx].pt for m in good_matches]
            # matched_keypoints2 = [unique_keypoints1[m.trainIdx].pt for m in good_matches]

            # Převod obrázku z BGR do RGB pro použití s Matplotlib
            image_rgb = cv2.cvtColor(gray1, cv2.COLOR_BGR2RGB)

            # Vykreslení bodů na obrázku
            for point in [kp.pt for kp in unique_keypoints1][:limit_of_points]:  # matched_keypoints1:
                plt.plot(point[0], point[1], 'ro', markersize=5)

            # Zobrazení obrázku s vykreslenými body
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            # Vykreslení shod na obrazu
            matched_image = cv2.drawMatches(gray1, selected_keypoints, gray2, keypoints2_sift, good_matches,
                                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Zobrazení výsledku
            cv2.namedWindow('Matched image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Matched image', np.int8(0.25 * width), np.int8(0.25 * height))
            cv2.imshow("Matched image", matched_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"\n\tHledání elementů dokončeno.\n\t\tDoba vytváření: {(time.time() - start_time):.2f} s.")

    outcome = np.zeros((len(x_old), 4))
    outcome[:, 0], outcome[:, 1], outcome[:, 2], outcome[:, 3] = x_old, y_old, x_new, y_new
    # outcome[:, 4], outcome[:, 5] = angle_old, angle_new

    return outcome, index


def circle_intersection(data):
    n = len(data)

    # Vytvořte matici koeficientů soustavy rovnic
    a = np.zeros((n - 1, 2))
    for i in range(1, n):
        x_i, y_i = data[i, 0], data[i, 1]
        x_1, y_1 = data[0, 0], data[0, 1]
        a[i - 1] = 2 * (x_i - x_1), 2 * (y_i - y_1)

    # Vytvořte vektor pravých stran soustavy rovnic
    b = np.zeros(n - 1)
    x_1, y_1 = data[0, 0], data[0, 1]
    r_1 = data[0, 2]
    for i in range(1, n):
        x_i, y_i = data[i, 0], data[i, 1]
        r_i = data[i, 2]
        b[i - 1] = (x_i ** 2 + y_i ** 2 - r_i ** 2) - (x_1 ** 2 + y_1 ** 2 - r_1 ** 2)

    # Vypočtěte souřadnice průsečíku
    intersection = np.linalg.lstsq(a, b, rcond=None)[0]

    return intersection


def distance_error(point, distances, known_points):
    x_i, y_i = point
    try:
        if len(known_points) == 0 and len(distances) == 0:
            return None
    except (ValueError, Exception):
        return None
    known_points = np.float64(known_points)
    distances = np.float64(distances)

    errors = (np.sqrt((x_i - known_points[:, 0]) ** 2 + (y_i - known_points[:, 1]) ** 2) - distances) ** 2
    error = np.sum(errors)
    return error


def second_circle_intersection(data, initial_guess):
    # if 'scipy.optimize' not in sys.modules:
    from scipy.optimize import least_squares
    # known_points = data[:, :2]
    # distances = data[:, 2]
    result = least_squares(distance_error, initial_guess, args=(data[:, 2], data[:, :2]))
    return result.x


def results_adjustment(result, old_center, limit, mesh, upper_area_cor=None):
    print("\nVyhodnocování výsledků.")

    new_center = np.empty((0, 2))

    tri_index = mesh.get_cells_type("triangle")

    triangle_points_old = mesh.points[:, :2]

    tri_cor_old = triangle_points_old[tri_index]
    tri_cor_new = np.empty((0, 3, 2))

    delete_indexes = []

    for i in range(len(limit) - 1):
        if limit[i] == limit[i + 1]:
            delete_indexes.append(i)
            continue

        segment = result[limit[i]:limit[i + 1], :4]
        size_segment = len(segment)

        """
        def calculate_angle(point0, points):
            vector = points - point0  # Rozdíl mezi prvním bodem a ostatními body
            angles = np.arctan2(vector[:, 1], vector[:, 0])  # Výpočet úhlu vzhledem k ose x
            return angles

        def rotation_matrix(angle):
            return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # Příklad použití
        original_points = segment[:, 0:2]
        original_center = old_center[i]
        new_points = segment[:, 2:4]

        original_angles = calculate_angle(original_points[0], original_points[1:])
        new_angles = calculate_angle(new_points[0], new_points[1:])

        new_position = [np.mean(new_points - original_points, axis=0)]
        a = [new_points[0] - original_points[0]]

        rotation_angle = np.mean(new_angles - original_angles)

        rot_pos = np.dot(original_center, rotation_matrix(rotation_angle))

        new_centers = new_position + rot_pos
        new_center = np.append(new_center, new_centers, axis=0)"""

        """if size_segment > 5:
            index_1, index_2 = np.int16(size_segment * 0.2), np.int16(size_segment * 0.4)
            index_3, index_4 = np.int16(size_segment * 0.6), np.int16(size_segment * 0.8)

        elif size_segment > 2:
            index_1 = 0
            index_2 = np.int16(size_segment * 0.5)
            index_3 = np.int16(size_segment * 0.5)
            index_4 = -1
        else:
            index_1 = index_2 = 0
            index_3 = index_4 = -1

        vector_members_old = segment[[0, index_1, index_2, index_3, index_4, -1]][:, [0, 1]]
        vector_members_new = segment[[0, index_1, index_2, index_3, index_4, -1]][:, [2, 3]]
        distances = np.linalg.norm(vector_members_old - old_center[i], axis=1)"""

        # Vypočet průsečíku a uložení
        try:
            matrix, r = cv2.findHomography(segment[:, 0:2].reshape(-1, 1, 2), segment[:, 2:4].reshape(-1, 1, 2),
                                           cv2.RANSAC, 5.0)
            success = np.sum(r) >= 0.75 * len(r)  # r.all() != 0 / r.all() == 1

        except cv2.error:
            success = False

        if success:
            transformed_data = cv2.perspectiveTransform(old_center[i].reshape(-1, 1, 2), matrix).reshape(-1, 2)
            new_center = np.append(new_center, transformed_data, axis=0)
        else:
            # Definice kružnic
            inputs = np.zeros((size_segment, 3))
            inputs[:, 0:2], inputs[:, 2], = segment[:, 2:4], np.linalg.norm(segment[:, :2] - old_center[i], axis=1)

            current_center = circle_intersection(inputs).reshape(1, 2)
            current_mean = np.mean(segment[:, 2:4], axis=0)
            calculation_type = np.linalg.norm(current_center - current_mean) < np.mean(inputs[:, 2])
            if calculation_type:
                new_center = np.append(new_center, current_center, axis=0)
            else:
                new_center = np.append(new_center,
                                       second_circle_intersection(inputs, current_mean).reshape(1, 2), axis=0)

        data = np.zeros((3, 2))
        for k in range(3):
            if success:
                data[k] = cv2.perspectiveTransform(tri_cor_old[i, k, :].reshape(-1, 1, 2), matrix).reshape(1, 2)
            else:
                # Definice kružnic
                inputs = np.zeros((size_segment, 3))
                inputs[:, 0:2] = segment[:, 2:4]
                inputs[:, 2] = np.linalg.norm(segment[:, :2] - tri_cor_old[i, k, :], axis=1)

                # Vypočet průsečíku a uložení
                if calculation_type:
                    data[k] = circle_intersection(inputs).reshape(1, 2)
                else:
                    data[k] = second_circle_intersection(inputs, current_mean).reshape(1, 2)
        tri_cor_new = np.append(tri_cor_new, [data], axis=0)

        if False:  # i == 40:
            original_points = segment[:, 0:2]
            original_center = old_center[i]
            new_points = segment[:, 2:4]

            plt.figure()
            fig.title("Centers of " + str(i) + ". element")
            plt.scatter(original_points[:, 0], original_points[:, 1], c='blue', marker='o', label='Původní body')
            plt.scatter(new_points[:, 0], new_points[:, 1], c='red', marker='o', label='Původní body')
            plt.scatter(original_center[0], original_center[1], c='green', marker='s', label='Původní těžiště')
            plt.scatter(new_center[i, 0], new_center[i, 1], c='black', marker='s', label='Nové těžiště')
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.show()

        # print(i, ":", limit[i + 1] - limit[i])

    tri_index = np.delete(tri_index, delete_indexes, axis=0)
    del delete_indexes

    """upper_area_elements = np.sum(old_center[:, 1] < 900)
    upper_area_lines = np.sum(result[:, 1] < 900)
    print("\n\tPočet horních elementů:", upper_area_elements, " , Počet řádků pro horní elementy", upper_area_lines)"""

    sx_old, sy_old, sx_new, sy_new = np.float64([]), np.float64([]), np.float64([]), np.float64([])
    # rot_x, rot_y = np.float64([]), np.float64([])

    for i in range(len(limit) - 1):
        if limit[i] == limit[i + 1]:
            continue

        segment = result[limit[i]:limit[i + 1]]
        sx_old = np.append(sx_old, np.mean(segment[:, 0], axis=0))
        sy_old = np.append(sy_old, np.mean(segment[:, 1], axis=0))
        sx_new = np.append(sx_new, np.mean(segment[:, 2], axis=0))
        sy_new = np.append(sy_new, np.mean(segment[:, 3], axis=0))

        """new_x = np.mean(segment[:, 2], axis=0) + (center[i, 0] - np.mean(segment[:, 0], axis=0))
        new_y = np.mean(segment[:, 3], axis=0) + (center[i, 1] - np.mean(segment[:, 1], axis=0))
        angle_rad = np.radians(result[i, 5]-result[i, 4])
        rot_x = np.append(rot_x, new_x * np.cos(angle_rad) - new_y * np.sin(angle_rad))
        rot_y = np.append(rot_y, new_x * np.sin(angle_rad) + new_y * np.cos(angle_rad))

    # s = np.stack((sx_old, sy_old, sx_new, sy_new), axis=1)"""

    print("\n\tÚspěšně nalezeno  {", len(new_center), "/", len(old_center), "}  elementů.")

    """# Vytvoříme prázdné pole pro nové vrcholy
    next_data1 = np.zeros_like(triangle_points_old)
    next_data1[tri_index] = tri_cor_new"""

    # Vytvoření nových trojúhelníků průměrem hodnot jednotlyvých elementů
    next_data = np.zeros_like(triangle_points_old, dtype=np.float64)
    np.add.at(next_data, np.int32(np.round(tri_index)), tri_cor_new)
    quantity = np.bincount(np.int32(np.round(tri_index)).ravel())

    # np.seterr(divide='ignore', invalid='ignore')  # ignorování chybových hlášek
    nonzero_indices = (quantity != 0)  # Kontrola na neplatné hodnoty před dělením
    try:
        next_data[nonzero_indices] /= quantity[nonzero_indices, np.newaxis]  # průměrování souřadnic počtem výskytu
    except IndexError:
        next_data = next_data[:len(quantity)]
        next_data[nonzero_indices] /= quantity[nonzero_indices, np.newaxis]
        print("\n\tChyba, seznam byl zkrácen.\n")
    # np.seterr(divide='warn', invalid='warn')  # obnovení chybových hlášek

    """wrong_list, wrong_index = [], []
    up_limit, down_limit, right_limit, left_limit = np.int32(min(points_pos[:, 1]) + 600), \
        np.int32(max(points_pos[:, 1]) + 50), np.int32(max(points_pos[:, 0]) + 600), 
        np.int32(min(points_pos[:, 0]) - 600)

    for t in range(len(new_center)):
        up_points = new_center[t, 1] < up_limit
        down_points = new_center[t, 1] > down_limit
        right_points = new_center[t, 0] > right_limit
        left_points = new_center[t, 0] < left_limit
        if up_points or down_points or right_points or left_points:
            wrong_index.append(t)
            wrong_list.append(limit[t + 1] - limit[t])
    if len(wrong_list) > 0:
        print("\n\tIndexy špatných bodů:", wrong_index)
        print("\tPočet špatných bodů:", wrong_list)
    else:
        del wrong_list, wrong_index
    del up_limit, down_limit, right_limit, left_limit"""

    dist = np.linalg.norm(segment[:, :2] - old_center[i], axis=1)

    difference_wrong = np.linalg.norm(new_center - np.array([sx_new, sy_new]).T, axis=1)
    # difference_wrong = np.linalg.norm(np.mean(new_center) - inputs[:, :2], axis=1)
    indices_wrong = np.where(np.mean(difference_wrong) >= np.mean(dist), 1, 0)
    indices_wrong = np.nonzero(indices_wrong)[0]
    distances_wrong = difference_wrong[indices_wrong]
    print("\n\tPočet špatných bodů:", len(indices_wrong),
          "\n\t\t-     Indexy:", indices_wrong,
          "\n\t\t- Vzdálenost:", distances_wrong)
    # print(new_center[indices_wrong])
    # print(np.array([sx_new, sy_new]).T[indices_wrong])

    if False:
        fig, ax = plt.subplots(figsize=(np.int8(0.0017 * width), np.int8(0.0017 * height)))
        plt.title("Triangle elements")
        plt.imshow(gray2, cmap='gray')

        plt.triplot(next_data[:, 0], next_data[:, 1], tri_index, color='green')

        plt.scatter(sx_new, sy_new, s=5, c='green', marker='s')

        plt.scatter(new_center[:, 0], new_center[:, 1], s=5, c='orange', marker='o')
        # Vykreslení obdélníka
        rec_w, rec_h = upper_area_cor[1, 0] - upper_area_cor[0, 0], upper_area_cor[1, 1] - upper_area_cor[0, 1]
        ax.add_patch(Rectangle((upper_area_cor[0]), rec_w, rec_h, edgecolor='firebrick', linewidth=1.5,
                               facecolor='none'))
        ax.add_patch(Rectangle((upper_area_cor[0]), rec_w, rec_h, facecolor='red', alpha=0.1))

        patches = [Polygon(triangle, closed=True, fill=None, color='royalblue') for triangle in tri_cor_new]
        ax.patches.extend(patches)

        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    if False:
        # fig, ax = plt.subplots(figsize=(np.int8(0.0017 * width), np.int8(0.0017 * height)))
        if state < 25:
            ax = plt.subplot(min(5, graph_count), min(5, graph_count), min(state + 1, 25))
        if state == 25:
            plt.pause(0.5)
            plt.figure(figsize=(15, 7))
            plt.pause(2)
        if state > 24:
            ax = plt.subplot(min(5, mgraph_count), min(5, graph_count), state - 24)

        plt.title('Triangle elements - Image {}.'.format(state + 1))
        # plt.title("Triangle elements")
        plt.imshow(gray2, cmap='gray')
        # plt.triplot(tri_cor_new[:, 0], tri_cor_new[:, 1], mesh.get_cells_type("triangle"))
        """patches = [Polygon(triangle, closed=True, fill=None, color='royalblue') for triangle in tri_cor_new]
        ax.patches.extend(patches)"""

        plt.triplot(next_data[:, 0], next_data[:, 1], tri_index, color='green')
        # plt.triplot(next_data1[:, 0], next_data1[:, 1], tri_index, color='red')

        plt.scatter(sx_new, sy_new, s=5, c='green', marker='s')
        plt.scatter(new_center[:, 0], new_center[:, 1], s=5, c='orange', marker='o')
        # Vykreslení obdélníka
        rec_w, rec_h = upper_area_cor[1, 0] - upper_area_cor[0, 0], upper_area_cor[1, 1] - upper_area_cor[0, 1]
        ax.add_patch(Rectangle((upper_area_cor[0]), rec_w, rec_h, edgecolor='firebrick', linewidth=1.5,
                               facecolor='none'))
        ax.add_patch(Rectangle((upper_area_cor[0]), rec_w, rec_h, facecolor='red', alpha=0.1))

        plt.axis('equal')
        plt.tight_layout()

        # Pauza pro zobrazení grafu
        plt.pause(0.5)
        if state == len(image_files) - 1:
            plt.show(block=True)
        else:
            plt.show(block=False)
        plt.pause(2)

        plt.close()

    if False:
        # Vykreslení těžišť - původní a nové nalezené
        plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.0017 * height)))
        plt.title("New and old element centers")
        plt.subplot(1, 2, 1)
        plt.imshow(gray1, cmap='gray')
        plt.scatter(old_center[:, 0], old_center[:, 1], s=20, c='orange', marker='o')

        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.imshow(gray2, cmap='gray')
        plt.scatter(new_center[:, 0], new_center[:, 1], s=20, c='orange', marker='o')

        plt.tight_layout()

        plt.show()

        print("Výsledky vhodnoceny.")

    return next_data, new_center, tri_index, np.array(indices_wrong, dtype=np.uint64)  # , tri_cor_new


def correlation_calculation():
    global points_pos, points_neg, points_cor, points_max, correlation_area_points_all, current_path_to_photos, \
        gray1, gray2, width, height

    if calculations_statuses['Correlation'] and not recalculate['Re Correlation']:
        return

    correlation_area_points_all = []

    print("\n\033[32m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
          "\n~~~~~~~~~~~~~~~~~          Correlation          ~~~~~~~~~~~~~~~~~"
          "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\033[0m")

    tot_im = len(image_files)
    for i in range(0, tot_im):
        print("\n=================================================================",
              f"\nAktuální proces:  [ {i + 1} / {tot_im} ]\t  Fotografie: {image_files[i]}")

        gray2 = load_photo(img_index=i, color_type=photo_type)

        correlation_area_points = []
        for j in range(len(points_cor)):
            text = f"\tOblast {j + 1}  [{j + 1} / {len(points_cor)}]"
            correlation_area_points.append(pixel_correlation(points_cor[j], text, min_length_x=200, min_length_y=50))

        correlation_area_points_all.append(correlation_area_points)  # n,2

    calculations_statuses['Correlation'] = True

    if make_temporary_savings:
        try_save_data(f'{saved_data_name}_temp', temporary_file=True)


def rough_calculation(mesh, centers):
    if calculations_statuses['Rough detection'] and not recalculate['Re Rough detection']:
        return

    global gray1, gray2, width, height
    global triangle_vertices_all, triangle_centers_all, triangle_indexes_all, triangle_points_all, \
        correlation_area_points_all, wrong_points_indexes_all, key_points_all, end_marks_all, current_path_to_photos
    global points_pos, points_neg, points_cor, points_max
    global set_n_features, set_n_octave_layers, set_contrast_threshold, set_edge_threshold, set_sigma

    key_points_all, end_marks_all, triangle_vertices_all, triangle_centers_all, triangle_indexes_all, \
        wrong_points_indexes_all = [], [], [], [], [], []

    print("\n\033[32m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
          "\n~~~~~~~~~~~~~~~~~        Rough detection        ~~~~~~~~~~~~~~~~~"
          "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\033[0m")

    if mesh is None or centers is None:
        print("Chyba v zadání")
        return

    tot_im = len(image_files)
    tot_roi = len(mesh)
    for i in range(0, tot_im):
        start__time = time.time()
        print("\n=================================================================",
              f"\nAktuální proces:  [ {i + 1} / {tot_im} ]\t  Fotografie: {image_files[i]}")

        gray2 = load_photo(img_index=i, color_type=photo_type)

        (temp_triangle_vertices, temp_triangle_centers, temp_triangle_indexes, temp_wrong_points_indexes,
         temp_key_points, temp_end_marks) = (
            np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64), np.empty((0, 3), dtype=np.uint64),
            np.empty(0, dtype=np.uint64), np.empty((0, 4), dtype=np.float64), [0])

        for j, current_mesh in enumerate(mesh):
            print("\n--------------------------------------------------------------",
                  f"\nAktuální ROI:  [ {j + 1} / {tot_roi} ]")

            try:
                tri_index_add = int(np.max(temp_triangle_indexes)) + 1
            except ValueError:
                tri_index_add = 0
            try:
                end_index_add = int(np.max(temp_end_marks))
            except ValueError:
                end_index_add = 0
            try:
                wrong_index_add = int(np.max(temp_wrong_points_indexes)) + 1
            except ValueError:
                wrong_index_add = 0

            key_points, end_marks = point_locator(mesh=current_mesh, shift_start=points_max, state=i,
                                                  current_shift=(correlation_area_points_all[i][0][0, 1] -
                                                                 points_cor[0][0, 1]),
                                                  n_features=set_n_features,
                                                  n_octave_layers=set_n_octave_layers,
                                                  contrast_threshold=set_contrast_threshold,
                                                  edge_threshold=set_edge_threshold,
                                                  sigma=set_sigma)

            triangle_vertices, triangle_centers, triangle_indexes, wrong_points_indexes = results_adjustment(
                key_points, centers[j], end_marks, current_mesh, correlation_area_points_all[i])

            triangle_indexes = triangle_indexes + tri_index_add
            if len(wrong_points_indexes) > 0:
                wrong_points_indexes = wrong_points_indexes + wrong_index_add

            temp_triangle_vertices = np.vstack((temp_triangle_vertices, triangle_vertices))
            temp_triangle_centers = np.vstack((temp_triangle_centers, triangle_centers))
            temp_triangle_indexes = np.vstack((temp_triangle_indexes, triangle_indexes))
            temp_wrong_points_indexes = np.hstack((temp_wrong_points_indexes, wrong_points_indexes))
            temp_key_points = np.vstack((temp_key_points, key_points))
            temp_end_marks.extend([e + end_index_add for e in end_marks[1:]])

        triangle_vertices_all.append(temp_triangle_vertices)  # n,2
        triangle_centers_all.append(temp_triangle_centers)  # n,2
        triangle_indexes_all.append(temp_triangle_indexes)  # n,3
        wrong_points_indexes_all.append(temp_wrong_points_indexes)  # n
        key_points_all.append(temp_key_points)  # n,4 => x_old, y_old, x_new, y_new
        end_marks_all.append(temp_end_marks)  # n

        print(f"\tCelkový čas: {time.time() - start__time:.2f}")

    # TODO udělat tohle rovnou ve funkci RESULTS
    triangle_points_all = [triangle_vertices_all[i][triangle_indexes_all[i]] for i in range(tot_im)]

    """if 'scipy.optimize' in sys.modules:
        del sys.modules['scipy.optimize']"""

    calculations_statuses['Rough detection'] = True

    if make_temporary_savings:
        try_save_data(f'{saved_data_name}_temp', temporary_file=True)


"""def fine_calculation():
    points_to_track = np.array([[2520, 1929], [1929, 2505], [2505, 3648], [3648, 4225], [4225, 3650], [3650, 3185],
                                [3185, 2969], [2969, 2105]])

    # Vytvoření prázdných mask pro obě oblasti
    mask1 = np.zeros(gray1.shape[:2], dtype=np.uint8)
    triangle_points = triangle_points_all[0]
    triangle_coordinates = triangle_points[1]
    cv2.fillPoly(mask1, [np.array(triangle_coordinates, dtype=np.int32)], 255)
    masked_image1 = gray1 & mask1

    mask2 = np.zeros(gray1.shape[:2], dtype=np.uint8)
    triangle_points = triangle_points_all[1]
    triangle_coordinates = triangle_points[1]
    cv2.fillPoly(mask2, [np.array(triangle_coordinates, dtype=np.int32)], 255)
    masked_image2 = gray2 & mask2

    points_to_track = np.mean(triangle_points[0], axis=0).reshape(1, 2)

    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('1', np.int8(0.1 * width), np.int8(0.1 * height))
    cv2.resizeWindow('2', np.int8(0.1 * width), np.int8(0.1 * height))
    cv2.imshow('1', masked_image1), cv2.imshow('2', masked_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Převod souřadnic bodů na formát akceptovaný funkcí cv2.goodFeaturesToTrack
    p0 = np.array(points_to_track, dtype=np.float32).reshape(-1, 1, 2)

    # Nastavení parametrů pro sledování
    lk_params = dict(winSize=(10, 10),
                     maxLevel=20,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

    # Výpočet optického toku
    p1, st, err = cv2.calcOpticalFlowPyrLK(masked_image2, masked_image2, p0, None, **lk_params)

    plot_point_size = 5

    # Výběr pouze bodů
    point_new = p1[st == 1]
    point_old = p0[st == 1]

    x_new = []
    y_new = []

    # Vykreslení sledovaných bodů a jejich trajektorie
    for j in range(2):
        frame = [image1, image2]
        for i, (new, old) in enumerate(zip(point_new, point_old)):
            a, b = new.ravel().astype(np.int32)
            c, d = old.ravel().astype(np.int32)
            frame[j] = cv2.line(frame[j], (a, b), (c, d), (0, 255, 0), plot_point_size)
            frame[j] = cv2.circle(frame[j], (a, b), 5, (0, 0, 255), plot_point_size * 3)
            frame[j] = cv2.circle(frame[j], (c, d), 5, (0, 0, 255), plot_point_size * 3)
            if j == 0:
                x_new = np.append(x_new, new[0])
                y_new = np.append(y_new, new[1])
        # cv2.rectangle(frame[j], (2650, 400), (3450, 650), (255, 255, 255), 3)

    window_name_1 = 'Fotografie 1:'
    window_name_2 = 'Fotografie 2:'

    # Vytvoření okna s rozměry odpovídajícími oříznutému výřezu
    cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(window_name_1, np.int8(0.25 * width), np.int8(0.25 * height))
    cv2.resizeWindow(window_name_2, np.int8(0.25 * width), np.int8(0.25 * height))

    # Zobrazení výsledného snímku s vykreslenými body a trajektoriemi
    cv2.imshow(window_name_1, image1)
    cv2.imshow(window_name_2, image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

"""
for i in range(len(image_files)):
    current_photo_fine_tri_points, current_photo_fine_centers = np.empty((0, 3, 2)), np.empty((0, 2))
    for j in range(len(triangle_points)):
        current_end_marks_all = end_marks_all[i]
        current_key_points = key_points_all[i][current_end_marks_all[j]:current_end_marks_all[j + 1]]

        fine_mesh, fine_mesh_centers = divide_image(triangle_points[j], mesh_size=mesh_size,
                                                    show_graph=False, printout=False)

        fine_triangle_points = fine_mesh.points[fine_mesh.get_cells_type("triangle")][:, :, :2]

        for z in range(len(fine_mesh_centers)):
            distances = np.sqrt((current_key_points[:, 0] - fine_mesh_centers[z, 0]) ** 2 +
                                (current_key_points[:, 1] - fine_mesh_centers[z, 1]) ** 2)

            radius = 10
            while True:
                points_in_radius = current_key_points[distances <= radius]
                if len(points_in_radius) > 3:
                    break
                else:
                    radius += 5

            M, _ = cv2.findHomography(points_in_radius[:, :2], points_in_radius[:, 2:], cv2.RANSAC, 5.0)

            data = np.zeros((3, 2))
            for k in range(3):
                try:
                    data[k] = cv2.perspectiveTransform(fine_triangle_points[z,
                    k].reshape(-1, 1, 2), M).reshape(1, 2)
                except cv2.error:
                    print([distances <= radius])
                    program_shutdown()
                current_photo_fine_tri_points = np.append(current_photo_fine_tri_points, [data], axis=0)

            transformed_data = cv2.perspectiveTransform(fine_mesh_centers.reshape(-1, 1, 2), M).reshape(-1, 2)
            current_photo_fine_centers = np.append(current_photo_fine_centers, transformed_data, axis=0)
"""


def scale_object(coordinates, scale_factor, axis: int | str = "vertical", only_x_cor=False, only_y_cor=False):
    # triangle_center = tuple(sum(x) / 3 for x in zip(*coordinates))  # Výpočet středu trojúhelníka
    if only_x_cor and only_y_cor:
        only_x_cor = only_y_cor = False

    if axis == 0 or axis == "vertical":
        pass
    elif axis == 1 or axis == "horizontal":
        coordinates = coordinates.T
    else:
        print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatně zvolený směr souřadnic objektu.")
        return

    center = np.mean(coordinates, axis=0)

    # Posunutí bodů polygonu tak, aby střed byl v počátku souřadnicového systému
    # new_coordinates = [(x - center[0], y - center[1]) for x, y in coordinates]

    # Zvětšení polygonu změnou měřítka
    # new_coordinates = [(scale_factor * x, scale_factor * y) for x, y in new_coordinates]

    """# Vrácení bodů na jejich původní místo (posunutí zpět)
    new_coordinates = np.float64([(x + center[0], y + center[1]) for x, y in
                                  [(scale_factor * x, scale_factor * y) for x, y in
                                   [(x - center[0], y - center[1]) for x, y in coordinates]]])"""

    new_coordinates = center + scale_factor * (coordinates.copy() - center)
    if only_x_cor:
        new_coordinates[:, 1] = coordinates[:, 1]
    elif only_y_cor:
        new_coordinates[:, 0] = coordinates[:, 0]

    if axis == 1 or axis == "horizontal":
        new_coordinates = new_coordinates.T

    return new_coordinates  # , center


def fast_fine_calculation(mesh_size=10):
    if calculations_statuses['Fine detection'] and not recalculate['Re Fine detection']:
        return

    global fine_triangle_points_all, fine_mesh_centers_all

    print("\n\033[32m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
          "\n~~~~~~~~~~~~~~~~~        Fine detection         ~~~~~~~~~~~~~~~~~"
          "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\033[0m")

    print("\nVytvoření podorobných elementů.")

    tot_im = len(image_files)

    def calculate_fine_mesh():
        fine_points_all, fine_centers_all = [], []
        triangle_points = triangle_points_all[0]

        for i in range(tot_im):
            current_photo_fine_points, current_photo_fine_centers = np.empty((0, 3, 2)), np.empty((0, 2))
            for j in range(len(triangle_points)):
                current_end_marks_all = end_marks_all[i]
                current_key_points = key_points_all[i][current_end_marks_all[j]:current_end_marks_all[j + 1]]

                mat, _ = cv2.findHomography(current_key_points[:, :2], current_key_points[:, 2:], cv2.RANSAC, 5.0)

                fine_mesh, fine_mesh_centers = divide_image(triangle_points[j], mesh_size=mesh_size,
                                                            show_graph=False, printout=False)

                fine_triangle_points = fine_mesh.points[fine_mesh.get_cells_type("triangle")][:, :, :2]

                data = np.zeros((3, 2))
                for p in range(len(fine_triangle_points)):
                    for k in range(3):
                        data[k] = cv2.perspectiveTransform(
                            fine_triangle_points[p, k].reshape(-1, 1, 2), mat).reshape(1, 2)
                    current_photo_fine_points = np.append(current_photo_fine_points, [data], axis=0)

                transformed_data = cv2.perspectiveTransform(fine_mesh_centers.reshape(-1, 1, 2), mat).reshape(-1, 2)
                current_photo_fine_centers = np.append(current_photo_fine_centers, transformed_data, axis=0)

            fine_points_all.append(current_photo_fine_points)
            fine_centers_all.append(current_photo_fine_centers)

        return fine_points_all, fine_centers_all

    fine_triangle_points_all, fine_mesh_centers_all = calculate_fine_mesh()

    calculations_statuses['Fine detection'] = True

    if make_temporary_savings:
        try_save_data(f'{saved_data_name}_temp', temporary_file=True)

    for h in ():  # range(len(image_files)):  # h = -1  # TODO KONTROLA #########
        plt.figure(num="Fine elements graph")
        gray_im = load_photo(img_index=h, color_type=photo_type)
        plt.imshow(gray_im, cmap='gray')

        [plt.gca().add_patch(Polygon(np.array(polygon_coords), edgecolor='b', facecolor='none'))
         for polygon_coords in fine_triangle_points_all[h]]

        triangles = triangle_vertices_all[h]
        tri_index = triangle_indexes_all[h]
        plt.triplot(triangles[:, 0], triangles[:, 1], tri_index, color='green')

        """patches = [Polygon(triangle, closed=True, fill=None, color='royalblue') for triangle in
                   fine_triangle_points_all[h]]
        for patch in patches:
            plt.gca().add_patch(patch)"""
        """plt.gca().add_patch(patches)"""

        """plt.scatter(fine_mesh_centers_all[h][:, 0], fine_mesh_centers_all[h][:, 1],
                    s=2,  # Velikost bodů
                    c="r",
                    marker='o')  # Znak bodů"""

        plt.tight_layout()
        plt.gca().autoscale(True)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.pause(0.5)
        plt.show(block=block_graphs)
        plt.pause(2)

    print("\nJemné prohledávání není zcela implementováno.")


def fine_calculation2(mesh_size=10):
    if calculations_statuses['Fine detection'] and not recalculate['Re Fine detection']:
        return

    global fine_triangle_points_all, fine_mesh_centers_all

    print("\nVytvoření podorobných elementů 2.")

    fine_triangle_points_all, fine_mesh_centers_all = [], []
    tot_im = len(image_files)

    def calculate_fine_mesh():
        current_photo_fine_tri_points, current_photo_fine_centers = None, None

        print()  # ############################################### TODO #######

        for i in range(tot_im):
            current_photo_fine_tri_points, current_photo_fine_centers = np.empty((0, 3, 2)), np.empty((0, 2))
            for j in range(len(triangle_points_all[0])):

                print("\tElement:", j + 1)

                transformation_matrix = cv2.getAffineTransform(np.float32(triangle_points_all[0][j]),
                                                               np.float32(triangle_points_all[i][j]))

                picture2 = load_photo(i, photo_type)
                h_, w_ = picture2.shape[:2]
                rotated_image = cv2.warpAffine(gray1, transformation_matrix, (w_, h_))

                fine_mesh, fine_mesh_centers = divide_image(triangle_points_all[0][j], mesh_size=mesh_size,
                                                            show_graph=False, printout=False)
                points_2 = cv2.transform(triangle_points_all[i][j].reshape(1, -1, 2),
                                         transformation_matrix).reshape(-1, 2)
                points_2 = np.int32(np.round(scale_object(coordinates=np.int32(np.round(points_2)),
                                                          scale_factor=1.5)))

                fine_triangle_points = fine_mesh.points[fine_mesh.get_cells_type("triangle")][:, :, :2]

                from templ_corr import do_templ_cor

                for fine_points in fine_triangle_points:
                    picture1 = gray1.copy()
                    picture2 = rotated_image.copy()
                    points = np.int32(np.round(fine_points))
                    x1, y1, w1, h1 = cv2.boundingRect(points)
                    x2, y2, w2, h2 = cv2.boundingRect(points_2)
                    points = points - np.array((x1, y1))
                    (x_bound, y_bound, w_bound, h_bound, points, picture1, pic1, pixel_values_np, min_position,
                     relative_points, picture2) = do_templ_cor(picture1=picture1, picture2=picture2, start1x=x1,
                                                               end1x=x1 + w1, start1y=y1, end1y=y1 + h1, start2x=x2,
                                                               end2x=x2 + w2, start2y=y2, end2y=y2 + h2, points=points,
                                                               cv2=cv2, np=np)

                    """plt.figure()
                    plt.subplot(231)
                    plt.scatter(x_bound, y_bound, color="yellowgreen", marker="+")
                    x_points, y_points = zip(*points)
                    plt.plot(x_points + (x_points[0],), y_points + (y_points[0],), linewidth=1, color='lime')

                    plt.imshow(cv2.cvtColor(picture1, cv2.COLOR_BGR2RGB))
                    rectangle = plt.Rectangle((x_bound, y_bound), w_bound, h_bound, edgecolor='green', facecolor='none')
                    plt.gca().add_patch(rectangle)

                    plt.subplot(232)
                    plt.imshow(cv2.cvtColor(pic1.astype(np.uint8), cv2.COLOR_BGR2RGB))

                    plt.subplot(233)
                    plt.imshow(pixel_values_np, cmap='jet', interpolation='nearest')
                    plt.colorbar()

                    plt.subplot(234)
                    plt.scatter(min_position[0], min_position[1], color="blue", marker="+")
                    x_points, y_points = zip(*(relative_points + min_position))
                    plt.plot(x_points + (x_points[0],), y_points + (y_points[0],), linewidth=1, color='skyblue')
                    plt.imshow(cv2.cvtColor(picture2, cv2.COLOR_BGR2RGB))
                    rectangle = plt.Rectangle((min_position[0], min_position[1]), w_bound, h_bound,
                                              edgecolor='dodgerblue',
                                              facecolor='none')
                    plt.gca().add_patch(rectangle)
                    plt.subplot(235)

                    mask2 = np.zeros((h_bound, w_bound), dtype=np.uint8)
                    pic2 = picture2[min_position[1]:min_position[1] + h_bound,
                           min_position[0]:min_position[0] + w_bound]
                    cv2.fillPoly(mask2, [relative_points], 255)
                    pic2 = pic2 & mask2
                    plt.imshow(cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB))

                    plt.subplot(236)
                    plt.imshow(cv2.cvtColor(picture2, cv2.COLOR_BGR2RGB))
                    x_points, y_points = zip(*(points_2 - np.array((x2, y2))).tolist())
                    plt.plot(x_points + (x_points[0],), y_points + (y_points[0],), linewidth=1, color='firebrick')
                    x_points, y_points = zip(*(triangle_points_all[i][j] - np.array((x2, y2))).tolist())
                    plt.plot(x_points + (x_points[0],), y_points + (y_points[0],), linewidth=1, color='coral')

                    plt.tight_layout()
                    plt.show()"""

                    found_points = relative_points + min_position + np.array((x2, y2))
                    found_center = np.mean(found_points, axis=0)

                    inv_trans_matrix = cv2.getAffineTransform(np.float32(triangle_points_all[i][j]),
                                                              np.float32(triangle_points_all[0][j]))

                    found_points_transformed = cv2.transform(found_points.reshape(1, -1, 2),
                                                             inv_trans_matrix).reshape(-1, 2)

                    found_center_transformed = cv2.transform(found_center.reshape(1, -1, 2),
                                                             inv_trans_matrix).reshape(2)

                    current_photo_fine_tri_points = np.append(current_photo_fine_tri_points,
                                                              [found_points_transformed], axis=0)
                    current_photo_fine_centers = np.append(current_photo_fine_centers,
                                                           [found_center_transformed], axis=0)

                if j == 3:  # teď smazat
                    return current_photo_fine_tri_points, current_photo_fine_centers

        return current_photo_fine_tri_points, current_photo_fine_centers

    fine_tr_point_cur, fine_mesh_point_cur = calculate_fine_mesh()
    fine_triangle_points_all.append(fine_tr_point_cur)
    fine_mesh_centers_all.append(fine_mesh_point_cur)

    calculations_statuses['Fine detection'] = True

    if make_temporary_savings:
        try_save_data(f'{saved_data_name}_temp', temporary_file=True)

    h = -1
    plt.figure()
    gray_im = load_photo(img_index=h, color_type=photo_type)
    plt.imshow(gray_im, cmap='gray')

    [plt.gca().add_patch(Polygon(np.array(polygon_coords), edgecolor='b', facecolor='none'))
     for polygon_coords in fine_triangle_points_all[h]]

    triangles = triangle_vertices_all[h]
    tri_index = triangle_indexes_all[h]
    plt.triplot(triangles[:, 0], triangles[:, 1], tri_index, color='green')

    """patches = [Polygon(triangle, closed=True, fill=None, color='royalblue') for triangle in
               fine_triangle_points_all[h]]
    for patch in patches:
        plt.gca().add_patch(patch)"""
    """plt.gca().add_patch(patches)"""

    """plt.scatter(fine_mesh_centers_all[h][:, 0], fine_mesh_centers_all[h][:, 1],
                s=2,  # Velikost bodů
                c="r",
                marker='o')  # Znak bodů"""

    plt.tight_layout()
    plt.gca().autoscale(True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.5)
    plt.show(block=block_graphs)
    plt.pause(2)

    print("\nJemné prohledávání 2 není zcela hotové.")


def fine_calculation_____________(mesh_size=10):
    if calculations_statuses['Fine detection'] and not recalculate['Re Fine detection']:
        return

    global fine_triangle_points_all, fine_mesh_centers_all

    print("\nVytvoření podorobných elementů 2.")

    fine_triangle_points_all, fine_mesh_centers_all = [], []

    def get_affine_matrix(a, b):
        # Výpočet afinní transformační matice mezi body A a B
        # Předpokládá se, že A a B jsou pole se dvěma body každým
        a_extended = np.vstack((a.T, np.ones((1, a.shape[0]))))
        b_extended = np.vstack((b.T, np.ones((1, b.shape[0]))))
        m, _ = np.linalg.lstsq(a_extended.T, b_extended.T, rcond=None)[:2]
        return m.T

    def transform_point(transformation_matrix, point):
        # Transformuje bod pomocí transformační matice.

        point_ = np.float32(point)  # TODO np.float64 - přesnost ????

        # Přidáme homogenní souřadnici (1) k bodu
        point_homog = np.append(point_, [1])

        # Použijeme afinní transformační matici na bod
        transformed_point_homog = np.dot(transformation_matrix, point_homog)

        # Převedeme homogenní souřadnice zpět na 2D souřadnice (x, y, 1) -> (x, y)
        transformed_point_1 = (transformed_point_homog[0], transformed_point_homog[1])

        # Transformuje bod pomocí transformační matice.

        point_ = np.float64(point)  # TODO np.float64 - přesnost ????

        # Přidáme homogenní souřadnici (1) k bodu
        point_homog = np.append(point_, [1])

        # Použijeme afinní transformační matici na bod
        transformed_point_homog = np.dot(transformation_matrix, point_homog)

        # Převedeme homogenní souřadnice zpět na 2D souřadnice (x, y, 1) -> (x, y)
        transformed_point_2 = (transformed_point_homog[0], transformed_point_homog[1])

        transformed__point = ((transformed_point_1[0] + transformed_point_2[0]) / 2,
                              (transformed_point_1[1] + transformed_point_2[1]) / 2)

        return transformed__point

    def calculate_fine_mesh():
        tot_im = len(image_files)

        for i in range(tot_im):
            current_photo_fine_tri_points, current_photo_fine_centers = np.empty((0, 3, 2)), np.empty((0, 2))
            for j in range(len(triangle_points_all[0])):

                transformation_matrix2 = cv2.getAffineTransform(np.float32(triangle_points_all[0][j]),
                                                                np.float32(triangle_points_all[i][j]))
                transformation_matrix = get_affine_matrix(triangle_points_all[0][j], triangle_points_all[i][j])

                picture2 = load_photo(i, photo_type)
                h_, w_ = picture2.shape[:2]
                rotated_image = cv2.warpPerspective(gray1, transformation_matrix, (w_, h_))
                rotated_image2 = cv2.warpAffine(gray1, transformation_matrix2, (w_, h_))

                input_points = np.float32([[10, 10], [20, 20], [30, 30]])

                # Aplikace afinní transformace
                input_points = cv2.transform(input_points.reshape(1, -1, 2), transformation_matrix2).reshape(-1, 2)

                fine_mesh, fine_mesh_centers = divide_image(triangle_points_all[0][j], mesh_size=mesh_size,
                                                            show_graph=False, printout=False)

                fine_triangle_points = fine_mesh.points[fine_mesh.get_cells_type("triangle")][:, :, :2]

                for p in range(len(fine_triangle_points)):
                    data = np.zeros((3, 2))
                    for k in range(3):
                        data[k] = transform_point(transformation_matrix, fine_triangle_points[p, k])
                        current_photo_fine_tri_points = np.append(current_photo_fine_tri_points, [data], axis=0)

                    """plt.figure()
                    plt.subplot(131)
                    plt.imshow(gray1)
                    plt.scatter(fine_triangle_points[p, :, 0], fine_triangle_points[p, :, 1], color="blue", marker="+")
                    plt.scatter(triangle_points_all[0][p, :, 0], triangle_points_all[0][p, :, 1], color="red",
                                marker="+")
                    plt.subplot(132)
                    plt.imshow(rotated_image)
                    point_to_transform = (2089.6251772501255, 2006.0266479862269)
                    transformed_point = transform_point(transformation_matrix, point_to_transform)
                    print(point_to_transform)
                    print(transformation_matrix)
                    print(transformed_point)
                    plt.scatter(data[0, 0], data[0, 1], color="red", marker="+")
                    plt.scatter(transformed_point[0], transformed_point[1])
                    plt.subplot(133)
                    plt.imshow(picture2)
                    plt.scatter(triangle_points_all[1][p, :, 0], triangle_points_all[1][p, :, 1], color="red",
                                marker="+")
                    plt.tight_layout()
                    plt.show()"""

                    points = np.int32(np.round(data))
                    x1, y1, w1, h1 = cv2.boundingRect(points)
                    points_2 = np.int32(np.round(
                        scale_object(coordinates=np.int32(np.round(triangle_points_all[i][j])), scale_factor=1.5)))
                    x2, y2, w2, h2 = cv2.boundingRect(points_2)
                    picture1 = gray1[y1:(y1 + h1), x1:(x1 + w1)]
                    picture2 = rotated_image[y2:(y2 + h2), x2:(x2 + w2)]
                    height1, width1 = picture1.shape[:2]
                    height2, width2 = picture2.shape[:2]
                    min_x, min_y = np.min(points[:, 0], axis=0), np.min(points[:, 1], axis=0)
                    points[:, 0], points[:, 1] = points[:, 0] - min_x, points[:, 1] - min_y

                    mask1 = np.zeros((height1, width1), dtype=np.uint8)
                    cv2.fillPoly(mask1, [points], 255)
                    pic1 = picture1 & mask1

                    x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(points)
                    relative_points = points - np.array([x_bound, y_bound])

                    pic1 = pic1[y_bound:(y_bound + h_bound), x_bound:(x_bound + w_bound)]

                    min1, max1 = np.min(pic1), np.max(pic1)
                    pic1 = cv2.normalize(pic1, None, min1, max1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)  # 0, 1

                    """for x in range(width2 - w_bound):
                        for y in range(height2 - h_bound):
                            mask2 = np.zeros((h_bound, w_bound), dtype=np.uint8)
                            pic2 = picture2[y:y + h_bound, x:x + w_bound]
                            cv2.fillPoly(mask2, [relative_points], 255)
                            pic2 = pic2 & mask2

                            pixel_values_np[y, x] = correlate(pic1, pic2)"""

                    pixel_values_np = np.float64([
                        [
                            np.linalg.norm(
                                pic1 ** 2 - cv2.normalize(picture2[y:y + h_bound, x:x + w_bound] & cv2.fillPoly(
                                    np.zeros((h_bound, w_bound), dtype=np.uint8), [relative_points], 255), None,
                                                          min1, max1, cv2.NORM_MINMAX, dtype=cv2.CV_64F) ** 2)
                            for x in range(width2 - w_bound)
                        ]
                        for y in range(height2 - h_bound)
                    ])

                    min_position = list(np.unravel_index(np.argmin(pixel_values_np), pixel_values_np.shape))
                    min_position[0], min_position[1] = min_position[1], min_position[0]
                    """print([x1, y1])
                    print(min_position)"""

                    if False:
                        plt.figure()
                        plt.subplot(141)
                        plt.imshow(picture1)
                        plt.subplot(143)
                        plt.imshow(picture2[min_position[1]:min_position[1] + h_bound,
                                   min_position[0]:min_position[0] + w_bound])
                        plt.subplot(144)
                        plt.imshow(picture2)
                        plt.gca().add_patch(Rectangle((min_position[0], min_position[1]), w_bound, h_bound,
                                                      edgecolor='red', facecolor='none'))
                        plt.tight_layout()
                        plt.show()

                transformed_data = cv2.perspectiveTransform(fine_mesh_centers.reshape(-1, 1, 2),
                                                            transformation_matrix).reshape(-1, 2)
                current_photo_fine_centers = np.append(current_photo_fine_centers, transformed_data, axis=0)

        return current_photo_fine_tri_points, current_photo_fine_centers

    fine_tr_point_cur, fine_mesh_point_cur = calculate_fine_mesh()
    fine_triangle_points_all.extend(fine_tr_point_cur)
    fine_mesh_centers_all.extend(fine_mesh_point_cur)

    calculations_statuses['Fine detection'] = True

    if make_temporary_savings:
        try_save_data(f'{saved_data_name}_temp', temporary_file=True)

    h = -1
    plt.figure()
    gray_im = load_photo(img_index=h, color_type=photo_type)
    plt.imshow(gray_im, cmap='gray')

    [plt.gca().add_patch(Polygon(np.array(polygon_coords), edgecolor='b', facecolor='none'))
     for polygon_coords in fine_triangle_points_all[h]]

    triangles = triangle_vertices_all[h]
    tri_index = triangle_indexes_all[h]
    plt.triplot(triangles[:, 0], triangles[:, 1], tri_index, color='green')

    """patches = [Polygon(triangle, closed=True, fill=None, color='royalblue') for triangle in
               fine_triangle_points_all[h]]
    for patch in patches:
        plt.gca().add_patch(patch)"""
    """plt.gca().add_patch(patches)"""

    """plt.scatter(fine_mesh_centers_all[h][:, 0], fine_mesh_centers_all[h][:, 1],
                s=2,  # Velikost bodů
                c="r",
                marker='o')  # Znak bodů"""

    plt.tight_layout()
    plt.gca().autoscale(True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.5)
    plt.show(block=block_graphs)
    plt.pause(2)

    print("\nJemné prohledávání 2 není.")


def finding_points():
    global points_pos, points_neg, points_cor, points_max, points_track
    global key_points_all, end_marks_all, tracked_points_all, tracked_rotations_all

    def is_inside_triangle(triangle, point):
        v0, v1, v2 = triangle
        x, y = point

        # Výpočet barycentrických souřadnic
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if denom != 0:
            alpha = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
            beta = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
            gamma = 1 - alpha - beta
        else:
            return False

        # Kontrola, zda bod leží uvnitř trojúhelníku
        if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
            return True
        else:
            return False

    print("\nSpuštěno hledaní bodů.")

    tracked_points_all = []
    tot_im = len(image_files)

    if 'points_track' not in globals() or (points_track is None or len(points_track) == 0):
        plt.close("Regions of interest")
        set_roi(finish_marking=True)

    gray_1 = load_photo(0, photo_type)
    for i in range(tot_im):
        gray_2 = load_photo(i, photo_type)

        if 'key_points_all' not in globals():
            for j in range(tot_im):
                global set_n_features, set_n_octave_layers, set_contrast_threshold, set_edge_threshold, set_sigma
                mesh = divide_image(points_pos, points_neg, size)[0]
                for m in mesh:
                    key_points, end_marks = point_locator(mesh=m, state=j,
                                                          n_features=set_n_features,
                                                          n_octave_layers=set_n_octave_layers,
                                                          contrast_threshold=set_contrast_threshold,
                                                          edge_threshold=set_edge_threshold,
                                                          sigma=set_sigma)
                    key_points_all.append(key_points)
                    end_marks_all.append(end_marks)

        for point_to_track, tracking_area in points_track[-1:]:  # TODO potom změnit na celý seznam
            # points_inside_polygon
            """current_key_points = [point for point in key_points_all[i][:, :2] if 
                                  Path(tracking_area).contains_point(point)]"""
            indexes = [index for index, point in enumerate(key_points_all[i][:, :2]) if
                       Path(tracking_area).contains_point(point)]
            current_key_points = key_points_all[i][indexes, :]
            print("\n")
            while len(current_key_points) < 5:
                print("\tSouřadnice zvolené hledané oblasti museli být zvětšeny.")
                tracking_area = scale_object(coordinates=tracking_area, scale_factor=1.1)
                indexes = [index for index, point in enumerate(key_points_all[i][:, :2]) if
                           Path(tracking_area).contains_point(point)]
                current_key_points = key_points_all[i][indexes, :]

            """current_key_points = key_points_all[i]
            distances = np.sqrt((current_key_points[:, 0] - point_to_track[0]) ** 2 +
                                (current_key_points[:, 1] - point_to_track[1]) ** 2)
            radius = 15
            while True:
                points_in_radius = current_key_points[distances <= radius]
                if len(points_in_radius) > 5:
                    break
                else:
                    radius += 2"""

            if True:
                plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)), num="Selected keypoints")
                plt.subplot(121)
                plt.title("Keypoints 1")
                plt.imshow(gray_1, cmap='gray')
                """plt.gca().add_patch(
                    plt.Circle(point_to_track, radius, edgecolor='blue', facecolor='royalblue', alpha=0.55))"""
                # plt.scatter(current_key_points[:, 0], current_key_points[:, 1], s=5, c='red', marker='o')
                plt.fill(tracking_area[:, 0], tracking_area[:, 1], facecolor='skyblue', edgecolor='none',
                         alpha=0.5)  # Vykreslení polygonu
                plt.scatter(current_key_points[:, 0], current_key_points[:, 1], s=10, c='orange', marker='o')
                plt.scatter(point_to_track[0], point_to_track[1], s=10, c='blue', marker='s')
                plt.axis('equal')

                plt.subplot(122)
                plt.title("Keypoints 2")
                plt.imshow(gray_2, cmap='gray')
                plt.scatter(current_key_points[:, 2], current_key_points[:, 3], s=10, c='orange', marker='o')
                plt.axis('equal')

                plt.tight_layout()
                plt.show()

            orig_pts = current_key_points[:, :2]
            def_pts = current_key_points[:, 2:]

            mask_1 = np.zeros(gray_1.shape[:2], dtype=np.uint8)
            mask_2 = mask_1.copy()

            [cv2.fillPoly(mask_1, [np.int32(np.round(polygon))], 255) for polygon in triangle_points_all[0]]
            masked_img_1 = cv2.bitwise_and(gray_1, gray_1, mask=mask_1)

            [cv2.fillPoly(mask_2, [np.int32(np.round(polygon))], 255) for polygon in triangle_points_all[i]]
            masked_img_2 = cv2.bitwise_and(gray_2, gray_2, mask=mask_2)

            window = 30

            mat, _ = cv2.findHomography(def_pts, orig_pts, cv2.RANSAC, 5.0)
            # M = cv2.getPerspectiveTransform(np.float32(def_pts), np.float32(orig_pts))

            x_max, x_min = np.int32(np.round(max(orig_pts[:, 0]) + window)), np.int32(
                np.round(min(orig_pts[:, 0]) - window))
            y_max, y_min = np.int32(np.round(max(orig_pts[:, 1]) + window)), np.int32(
                np.round(min(orig_pts[:, 1]) - window))

            # Zobrazení výsledku
            plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)), num="Cropped tracked area")
            plt.subplot(121)
            plt.imshow(masked_img_1[y_min:y_max, x_min:x_max], cmap='gray')
            plt.axis('equal')

            # Aplikace transformace na trojúhelníkovou oblast
            triangle_pts = np.float32([def_pts]).reshape(-1, 1, 2)
            transformed_pts = cv2.perspectiveTransform(triangle_pts, mat).reshape(triangle_pts.shape[0], -1)

            xt_max, xt_min = (np.int32(np.round(max(transformed_pts[:, 0]) + window)),
                              np.int32(np.round(min(transformed_pts[:, 0]) - window)))
            yt_max, yt_min = (np.int32(np.round(max(transformed_pts[:, 1]) + window)),
                              np.int32(np.round(min(transformed_pts[:, 1]) - window)))

            transformed_img = cv2.warpPerspective(masked_img_2, mat, (masked_img_2.shape[1], masked_img_2.shape[0]))

            plt.subplot(122)
            plt.imshow(transformed_img[yt_min:yt_max, xt_min:xt_max], cmap='gray')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

            side = 10

            px, py = point_to_track
            template = gray_1[py - side:py + side + 1, px - side:px + side + 1]
            result = cv2.matchTemplate(transformed_img[yt_min:yt_max, xt_min:xt_max], template,
                                       cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

            # Zobrazení výsledku
            plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)),
                       num="Cropped tracked area with center")
            plt.subplot(121)
            plt.imshow(gray_1[y_min:y_max, x_min:x_max], cmap='gray')
            plt.gca().add_patch(
                Rectangle((px - side - x_min, py - side - y_min), template.shape[1], template.shape[0],
                          edgecolor='crimson', facecolor='coral', alpha=0.2))
            plt.scatter(px - x_min, py - y_min, color='red', marker='x')
            plt.axis('equal')

            plt.subplot(122)
            plt.imshow(transformed_img[yt_min:yt_max, xt_min:xt_max], cmap='gray')
            plt.gca().add_patch(Rectangle((top_left[0], top_left[1]), template.shape[1], template.shape[0],
                                          edgecolor='crimson', facecolor='coral', alpha=0.2))
            plt.scatter(top_left[0] + side, top_left[1] + side, color='red', marker='x')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

            original_point = np.float32(point_to_track).reshape(-1, 1, 2)
            transformed_point = cv2.perspectiveTransform(original_point, mat)

            # Výpočet inverzní transformační matice
            inv_mat = cv2.invert(mat)[1]

            # Aplikace inverzní transformace na transformované souřadnice
            original_point__ = cv2.perspectiveTransform(np.float32(
                (top_left[0] + side + xt_min, top_left[1] + side + yt_min)).reshape(-1, 1, 2), inv_mat)[0][0]

            print("souřadnice bodu na původní fotce:\n", point_to_track)

            mat2, _ = cv2.findHomography(orig_pts, def_pts, cv2.RANSAC, 5.0)
            transformed_point2 = cv2.perspectiveTransform(original_point, mat2)[0][0]
            inv_mat2 = cv2.invert(mat2)[1]
            original_point_ = cv2.perspectiveTransform(transformed_point, inv_mat2)[0][0]
            print("souřadnice bodu na druhé fotce po transformaci:\n", transformed_point2)
            print("souřadnice předchozího bodu na druhé fotce po transformaci a inverzní transformaci:\n",
                  original_point_)
            print("souřadnice bodu na druhé transformované fotce:\n",
                  [top_left[0] + side + xt_min, top_left[1] + side + yt_min])
            print("souřadnice bodu na druhé transformované fotce po transformaci:\n", original_point__)

            x2_max, x2_min = np.int32(np.round(max(def_pts[:, 0]) + window)), np.int32(
                np.round(min(def_pts[:, 0]) - window))
            y2_max, y2_min = np.int32(np.round(max(def_pts[:, 1]) + window)), np.int32(
                np.round(min(def_pts[:, 1]) - window))

            mask1 = np.zeros(gray_1.shape[:2], dtype=np.uint8)
            mask2 = mask1.copy()

            cv2.rectangle(mask1, (x_min, y_min), (x_max, y_max), 255, -1)
            masked_image1 = gray_1 & mask1

            cv2.rectangle(mask2, (x2_min, y2_min), (x2_max, y2_max), 255, -1)
            masked_image2 = gray_2 & mask2

            p0 = np.float32(point_to_track).reshape(-1, 1, 2)

            """# Nastavení parametrů pro metodu goodFeaturesToTrack
            feature_params = dict(maxCorners=500,
                                  qualityLevel=0.7,
                                  minDistance=3,
                                  blockSize=2)  # citlivost -> větší = ignorace (detekte pixelů / šum)

            # Získání bodů v omezené oblasti pomocí goodFeaturesToTrack a masky
            p0 = cv2.goodFeaturesToTrack(gray_1, mask=mask1, **feature_params)"""

            # Nastavení parametrů pro sledování
            lk_params = dict(winSize=(10, 10),
                             maxLevel=5,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 0.01))

            # Výpočet optického toku
            p1, st, err = cv2.calcOpticalFlowPyrLK(masked_image1, masked_image2, p0, None, **lk_params)

            # Výběr pouze bodů
            point_old = p0[st == 1].reshape(-1, 2)
            point_new = p1[st == 1].reshape(-1, 2)

            # Zobrazení výsledku
            plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)), num="Cropped area with center "
                                                                                       "- OPT flow")
            plt.subplot(121)
            plt.imshow(masked_image1, cmap='gray')
            plt.scatter(point_old[:, 0], point_old[:, 1], color='red', marker='x')
            plt.axis('equal')

            plt.subplot(122)
            plt.imshow(masked_image2, cmap='gray')
            plt.scatter(point_new[:, 0], point_new[:, 1], color='red', marker='x')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

            mask1s = np.zeros(gray_1.shape[:2], dtype=np.uint8)
            mask2s = mask1s.copy()

            # Nakreslení kruhu (vyplnění bílým)
            cv2.circle(mask1s, (px, py), 75, 255, -1)
            masked_image1 = gray_1 & mask1s

            # Nakreslení kruhu (vyplnění bílým)
            cv2.circle(mask2s, (np.int32(np.round(np.squeeze(original_point__)[0])),
                                np.int32(np.round(np.squeeze(original_point__)[1]))), 75, 255, -1)
            masked_image2 = gray_2 & mask2s

            met = cv2.SIFT_create()
            keypoints1_s, descriptors1_s = met.detectAndCompute(gray_1, mask1s)
            keypoints2_s, descriptors2_s = met.detectAndCompute(gray_2, mask2s)

            # Zobrazení výsledku
            plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)),
                       num="Area with created keypoints")
            plt.subplot(121)
            plt.imshow(cv2.drawKeypoints(masked_image1, keypoints1_s, masked_image1), cmap='gray')
            plt.axis('equal')
            plt.subplot(122)
            plt.imshow(cv2.drawKeypoints(masked_image2, keypoints2_s, masked_image2), cmap='gray')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

            # Shoda klíčových bodů
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1_s, descriptors2_s, k=2)
            good_matches = [m for m, n in matches if m.distance < precision * n.distance]

            # Vykreslení shod na obrazu
            matched_image = cv2.drawMatches(gray_1, keypoints1_s, gray_2, keypoints2_s, good_matches, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.figure(num="Matched area keypoints")
            plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

            # Vybrání shodných klíčových bodů
            src_pts = np.float32([keypoints1_s[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints2_s[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # Výpočet transformační matice
            mat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            original_point = np.float32(point_to_track).reshape(-1, 1, 2)
            transformed_point = cv2.perspectiveTransform(original_point, mat).reshape(-1, 2)

            print("Transformovaný střed:\n", transformed_point)

            # Zobrazení výsledku
            plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)), num="Transformed center")
            plt.subplot(121)
            plt.imshow(gray_1, cmap='gray')
            plt.scatter(px, py, color='red', marker='x')
            plt.axis('equal')

            plt.subplot(122)
            plt.imshow(gray_2, cmap='gray')
            plt.scatter(transformed_point[:, 0], transformed_point[:, 1], color='red', marker='x')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()


def point_tracking_calculation(use_correlation=True, interpolate_new_points=False, interpolation_number=16):
    if calculations_statuses['Point detection'] and not recalculate['Re Point detection']:
        return

    global points_pos, points_neg, points_cor, points_max, points_track
    global triangle_points_all, key_points_all, end_marks_all, tracked_points_all, tracked_rotations_all

    print("\n\033[32m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
          "\n~~~~~~~~~~~~~~~~~        Point detection        ~~~~~~~~~~~~~~~~~"
          "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\033[0m")

    print("\nSpuštěno hledaní bodů.")

    show_graphs = False  # TODO UKAZKA GRAFU

    tracked_points_all, tracked_rotations_all = [], []

    if 'points_track' not in globals() or (points_track is None or len(points_track) == 0):
        plt.close("Regions of interest")
        set_roi(finish_marking=True)

    tot_im = len(image_files)

    if interpolate_new_points:
        interpolation_number = round(interpolation_number)
        new_points_track = []

        def linear_interpolation(point1, point2, num_interpolated_points, polygon, delete_first_last=False):
            t = np.linspace(0, 1, num_interpolated_points + 1)
            x = point1[0] + t * (point2[0] - point1[0])
            y = point1[1] + t * (point2[1] - point1[1])
            interpolated_points = np.column_stack((x, y))
            polygon_center = np.mean(polygon, axis=0)
            interpolated_polygon = np.array([polygon.copy() + (interpolated_points[m] - polygon_center)
                                             for m in range(num_interpolated_points + 1)])
            if delete_first_last:
                return interpolated_points[1:-1], interpolated_polygon[1:-1]
            else:
                return interpolated_points, interpolated_polygon

        area = points_track[0][1].copy()
        points_track_new = points_track[:6].copy()
        points_track_new.append(points_track_new[0])
        for i in range(len(points_track_new) - 1):
            new_points_track.append(points_track_new[i])
            inter_points = linear_interpolation(points_track_new[i][0], points_track_new[i + 1][0],
                                                interpolation_number, area, True)
            [new_points_track.append((inter_points[0][j], inter_points[1][j])) for j in range(len(inter_points[0]))]
            # new_points_track.append(points_track_new[i + 1])

        if len(points_track) == 10:
            new_points_track.append(points_track[6])
            inter_points = linear_interpolation(points_track[6][0], points_track[8][0], interpolation_number,
                                                area, True)
            [new_points_track.append((inter_points[0][j], inter_points[1][j])) for j in range(len(inter_points[0]))]
            new_points_track.append(points_track[8])

            new_points_track.append(points_track[7])
            inter_points = linear_interpolation(points_track[7][0], points_track[9][0], interpolation_number,
                                                area, True)
            [new_points_track.append((inter_points[0][j], inter_points[1][j])) for j in range(len(inter_points[0]))]
            new_points_track.append(points_track[9])

        points_track = new_points_track.copy()
        del new_points_track, points_track_new, inter_points, interpolation_number

    tot_p = len(points_track)

    if show_graphs:
        plt.figure(num="Tracked points")
        plt.imshow(cv2.cvtColor(load_photo(0, photo_type), cv2.COLOR_BGR2RGB))
        [plt.scatter(p[0][0], p[0][1], c='blue', zorder=3) for p in points_track]
        [plt.text(p[0][0] + 10, p[0][1] - 10, f"{n + 1}", fontsize=5, ha='left', va='bottom', color='darkblue',
                  fontweight='bold') for n, p in enumerate(points_track)]
        [plt.fill(p[1][:, 0], p[1][:, 1], facecolor='skyblue', edgecolor='none', alpha=0.5) for p in points_track]
        plt.show()

    if 'key_points_all' not in globals():
        for j in range(tot_im):
            global set_n_features, set_n_octave_layers, set_contrast_threshold, set_edge_threshold, set_sigma
            mesh = divide_image(points_pos, points_neg, size)[0]
            for m in mesh:
                key_points, end_marks = point_locator(mesh=m, state=j,
                                                      n_features=set_n_features,
                                                      n_octave_layers=set_n_octave_layers,
                                                      contrast_threshold=set_contrast_threshold,
                                                      edge_threshold=set_edge_threshold,
                                                      sigma=set_sigma)
                key_points_all.append(key_points)
                end_marks_all.append(end_marks)

    if use_correlation or show_graphs:
        gray_1 = load_photo(0, photo_type)
        mask_1 = np.zeros(gray_1.shape[:2], dtype=np.uint8)
        [cv2.fillPoly(mask_1, [np.int32(np.round(polygon))], 255) for polygon in triangle_points_all[0]]
        masked_img_1 = gray_1 & mask_1

    for i in range(tot_im):
        print("\n=================================================================",
              f"\nAktuální proces:  [ {i + 1} / {tot_im} ]\t  Fotografie: {image_files[i]}")

        start_time = time.time()
        if use_correlation or show_graphs:
            gray_2 = load_photo(i, photo_type)
        tracked_points_cur = np.empty((0, 2), dtype=np.int32)
        tracked_rotations = np.empty(0, dtype=np.float64)

        for j, (point_to_track, tracking_area) in enumerate(points_track):
            # points_inside_polygon
            indexes = [index for index, point in enumerate(key_points_all[i][:, :2]) if
                       Path(tracking_area).contains_point(point)]
            current_key_points = key_points_all[i][indexes, :]

            c = 1
            while len(current_key_points) < 5:
                tracking_area = scale_object(coordinates=tracking_area, scale_factor=1.05)
                indexes = [index for index, point in enumerate(key_points_all[i][:, :2]) if
                           Path(tracking_area).contains_point(point)]
                current_key_points = key_points_all[i][indexes, :]
                c += 1
            if c > 1:
                print(f"\n\t - Souřadnice zvolené hledané oblasti museli být zvětšeny {1.05 ** c: .2f} krát.\n")

            if show_graphs:
                plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)), num="Selected keypoints")
                plt.subplot(121)
                plt.title("Keypoints 1")
                plt.imshow(gray_1, cmap='gray')
                """plt.gca().add_patch(
                    plt.Circle(point_to_track, radius, edgecolor='blue', facecolor='royalblue', alpha=0.55))"""
                # plt.scatter(current_key_points[:, 0], current_key_points[:, 1], s=5, c='red', marker='o')
                plt.fill(tracking_area[:, 0], tracking_area[:, 1], facecolor='skyblue', edgecolor='none',
                         alpha=0.5)  # Vykreslení polygonu
                plt.scatter(current_key_points[:, 0], current_key_points[:, 1], s=10, c='orange', marker='o')
                plt.scatter(point_to_track[0], point_to_track[1], s=10, c='blue', marker='s')
                plt.axis('equal')

                plt.subplot(122)
                plt.title("Keypoints 2")
                plt.imshow(gray_2, cmap='gray')
                plt.scatter(current_key_points[:, 2], current_key_points[:, 3], s=10, c='orange', marker='o')
                plt.axis('equal')

                plt.tight_layout()
                plt.show()

            # orig_pts = current_key_points[:, :2]
            # def_pts = current_key_points[:, 2:]

            if use_correlation or show_graphs:
                mask_2 = np.zeros(gray_2.shape[:2], dtype=np.uint8)
                [cv2.fillPoly(mask_2, [np.int32(np.round(polygon))], 255) for polygon in triangle_points_all[i]]
                masked_img_2 = gray_2 & mask_2

                """# Morfologická operace dilatace pro zacelení děr a eroze pro odstranění malých objektů a zúžení hran
                kernel = np.ones((4, 4), np.uint8)
                mask_2 = cv2.erode(cv2.dilate(mask_2, kernel, iterations=2), kernel, iterations=1)
                masked_img_2 = cv2.bitwise_and(gray_2, gray_2, mask=mask_2)"""

            # Odhad transformační matice
            transform_matrix = cv2.estimateAffinePartial2D(current_key_points[:, 2:].astype(np.float32),
                                                           current_key_points[:, :2].astype(np.float32))[0]

            # Matice obsahuje informace o translaci (posunu) a rotaci
            rotation = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            tracked_rotations = np.append(tracked_rotations, rotation)

            tran_mat = cv2.findHomography(current_key_points[:, 2:], current_key_points[:, :2], cv2.RANSAC, 5.0)[0]
            # tran_mat = cv2.getPerspectiveTransform(np.float32(def_pts), np.float32(orig_pts))
            tran_mat_inv = cv2.findHomography(current_key_points[:, :2], current_key_points[:, 2:], cv2.RANSAC, 5.0)[0]
            # tran_mat_inv = cv2.invert(tran_mat)[1]

            if use_correlation or show_graphs:
                bound = np.int32(cv2.boundingRect(np.float32(tracking_area))).reshape(2, 2)
                w, h = bound[1] // 4
                bound[1] += bound[0]
                bound = np.int32(np.round(scale_object(coordinates=bound, scale_factor=2.5)))

                w, h = max(w, 20), max(h, 20)
                track_area = np.int32((point_to_track - (w, h), point_to_track + (w, h)))

                img_warped = cv2.warpPerspective(masked_img_2, tran_mat, (gray_2.shape[1], gray_2.shape[0]))

                template = masked_img_1[track_area[0, 1]:track_area[1, 1], track_area[0, 0]:track_area[1, 0]]

            if show_graphs:
                plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)), num="Areas")
                plt.subplot(221)
                plt.title("Image 1")
                plt.imshow(gray_1, cmap='gray')
                plt.fill(tracking_area[:, 0], tracking_area[:, 1], color='blue', alpha=0.5)  # Vykreslení polygonu
                orig_bound = np.int32(cv2.boundingRect(np.float32(tracking_area))).reshape(2, 2)
                plt.gca().add_patch(
                    Rectangle(orig_bound[0], orig_bound[1, 0], orig_bound[1, 1], color='green', alpha=0.5))
                plt.gca().add_patch(
                    Rectangle(track_area[0], template.shape[0], template.shape[1], color='orange', alpha=0.5))
                plt.scatter(point_to_track[0], point_to_track[1], s=10, c='red', marker='o')
                plt.axis('equal')

                plt.subplot(222)
                plt.title("Image 2")
                plt.imshow(img_warped, cmap='gray')
                plt.gca().add_patch(Rectangle(bound[0], bound[1, 0] - bound[0, 0], bound[1, 1] - bound[0, 1],
                                              color='orange', alpha=0.5))
                plt.axis('equal')

                plt.subplot(223)
                plt.title("Template")
                plt.imshow(template, cmap='gray')
                plt.axis('equal')

                plt.subplot(224)
                plt.title("Tracked image")
                plt.imshow(img_warped[bound[0, 1]:bound[1, 1], bound[0, 0]:bound[1, 0]], cmap='gray')
                plt.axis('equal')

                plt.tight_layout()
                plt.show()

            if use_correlation or show_graphs:
                result = cv2.matchTemplate(img_warped[bound[0, 1]:bound[1, 1], bound[0, 0]:bound[1, 0]],
                                           template, cv2.TM_CCOEFF_NORMED)
                top_left = cv2.minMaxLoc(result)[-1] + (point_to_track - track_area[0])

            if show_graphs:
                max_loc = cv2.minMaxLoc(result)[-1]
                # Zobrazení výsledku
                plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)),
                           num="Cropped tracked area with center")
                plt.subplot(221)
                plt.imshow(masked_img_1[bound[0, 1]:bound[1, 1], bound[0, 0]:bound[1, 0]], cmap='gray')
                plt.gca().add_patch(Rectangle((max_loc[0], max_loc[1]), template.shape[1], template.shape[0],
                                              edgecolor='crimson', facecolor='coral', alpha=0.2))
                plt.scatter(top_left[0], top_left[1], color='red', marker='x')
                plt.axis('equal')

                plt.subplot(222)
                plt.imshow(img_warped[bound[0, 1]:bound[1, 1], bound[0, 0]:bound[1, 0]], cmap='gray')
                plt.gca().add_patch(Rectangle((max_loc[0], max_loc[1]), template.shape[1], template.shape[0],
                                              edgecolor='green', facecolor='yellow', alpha=0.2))
                plt.scatter(top_left[0], top_left[1], color='greenyellow', marker='x')
                plt.axis('equal')

                plt.tight_layout()
                plt.show()

            transformed_point = cv2.perspectiveTransform(np.float32(point_to_track).reshape(-1, 1, 2),
                                                         tran_mat_inv)[0][0]
            if use_correlation:
                transformed_matched_point = cv2.perspectiveTransform(np.float32(top_left + bound[0]).reshape(-1, 1, 2),
                                                                     tran_mat_inv)[0][0]
                # result_point = np.int32(np.round((transformed_matched_point + transformed_point) / 2))
                result_point = np.int32(np.round(transformed_matched_point))
            else:
                result_point = np.int32(np.round(transformed_point))

            tracked_points_cur = np.append(tracked_points_cur, result_point.reshape(1, 2), axis=0)

            if show_graphs:
                matched_point = np.int32(top_left + bound[0])
                transformed_point = np.int32(np.round(transformed_point))
                transformed_matched_point = np.int32(
                    np.round(transformed_matched_point)) if use_correlation else transformed_point

                print("\nPůvodní hledaný bod (foto 1):\n\t", point_to_track)
                print("Hledaný bod (foto 2):\n\t", transformed_point)
                print("Matchnutý bod:", matched_point, "Transformovaný matched bod:", transformed_matched_point)
                print("Result bod:", result_point)
                print(point_to_track - matched_point, transformed_point - transformed_matched_point)

                # Zobrazení výsledku
                plt.figure(figsize=(np.int8(0.002 * width), np.int8(0.002 * height)), num="Final matched point")
                plt.subplot(121)
                plt.imshow(gray_1, cmap='gray')
                plt.scatter(point_to_track[0], point_to_track[1], s=10, c='blue', marker='s', zorder=3)
                plt.scatter(current_key_points[:, 0], current_key_points[:, 1], s=10, c='orange', marker='o')
                # Vytvoření grafu a nakreslení čáry
                plt.plot([point_to_track[0], point_to_track[0] + 200], [point_to_track[1], point_to_track[1]],
                         color='green', lw=1.5)
                plt.axis('equal')

                plt.subplot(122)
                plt.imshow(gray_2, cmap='gray')
                plt.scatter(transformed_matched_point[0], transformed_matched_point[1], s=10, color='blue', marker='s',
                            zorder=3)
                plt.scatter(current_key_points[:, 2], current_key_points[:, 3], s=10, c='orange', marker='o')
                points__ = np.array([[transformed_matched_point[0], transformed_matched_point[1]],
                                     [transformed_matched_point[0] + 200, transformed_matched_point[1]]])
                plt.plot(points__[:, 0], points__[:, 1], color='red', lw=4)
                rot_mat = cv2.getRotationMatrix2D(transformed_matched_point.tolist(), np.rad2deg(rotation), 1.0)
                points__ = cv2.transform(points__.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
                plt.plot(points__[:, 0], points__[:, 1], color='green', lw=1.5)

                plt.axis('equal')
                plt.tight_layout()
                plt.show()
            else:
                print_progress_bar(j + 1, tot_p, 1, 20, "\t")

        tracked_points_all.append(tracked_points_cur)
        tracked_rotations_all.append(tracked_rotations)
        print(f"\n\tDoba vytváření: {time.time() - start_time: .2f} s.")

    calculations_statuses['Point detection'] = True

    if make_temporary_savings:
        try_save_data(f'{saved_data_name}_temp', temporary_file=True)


"""triangle_points1 = triangle_points_all[0]
mask1 = np.zeros(gray1.shape[:2], dtype=np.uint8)
mask2 = mask1.copy()

j = 1  # která fotka
i = 1  # který trojúhelník

triangle_coordinates1 = triangle_points1[i]
cv2.fillPoly(mask1, [np.int32(triangle_coordinates1)], 255)
masked_image1 = gray1 & mask1

triangle_points2 = triangle_points_all[j]
triangle_coordinates2 = triangle_points2[i]
cv2.fillPoly(mask2, [np.int32(triangle_coordinates2)], 255)
masked_image2 = gray2 & mask2"""

"""mask = np.zeros(gray1.shape[:2], dtype=np.uint8)
p = np.array(((points_cor[0]), (points_cor[0, 0], points_cor[1, 1]), (points_cor[1]), 
(points_cor[1, 0], points_cor[0, 0])), np.int32)
cv2.fillPoly(mask, [p], 255)
t = (gray1 & mask)[350:700, 2550:3500]

plt.figure()
plt.imshow(t)
plt.show()

result = cv2.matchTemplate(gray1, t, cv2.TM_SQDIFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + t.shape[1], top_left[1] + t.shape[0])

print(top_left, bottom_right)

plt.figure()
cv2.rectangle(image1, top_left, bottom_right, (0, 0, 255), 10)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.show()"""

"""# Načtěte původní a cílovou fotografii
img1 = gray1
img2 = gray2

# Detekce klíčových bodů a výpočet deskriptorů
orb = cv2.SIFT_create()  # ####################### #TODO #################################### Změnit na Sift
keypoints1_sift, descriptors1_sift = orb.detectAndCompute(img1, mask1)
keypoints2_sift, descriptors2_sift = orb.detectAndCompute(img2, mask2)

if True:
    plt.figure(figsize=(np.int8(0.0017 * width), np.int8(0.001 * height)))
    plt.subplot(121)
    plt.title("Keypoints 1")
    plt.imshow(cv2.drawKeypoints(img1, keypoints1_sift, img1), cmap='gray')
    plt.axis('equal')

    plt.subplot(122)
    plt.title("Keypoints 2")
    plt.imshow(cv2.drawKeypoints(img2, keypoints2_sift, img2), cmap='gray')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Shoda klíčových bodů
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1_sift, descriptors2_sift, k=2)  # ORB 141, 56
good_matches = [m for m, n in matches if m.distance < precision * n.distance]  # ORB 36, 40

# Vybrání shodných klíčových bodů
src_pts = np.float32([keypoints1_sift[match.queryIdx].pt for match in good_matches]).reshape(-1, 2)
dst_pts = np.float32([keypoints2_sift[match.trainIdx].pt for match in good_matches]).reshape(-1, 2)

# Výpočet transformační matice
M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

x_max, x_min = np.int32(max(src_pts[:, 0]) + 50), np.int32(min(src_pts[:, 0]) - 50)
y_max, y_min = np.int32(max(src_pts[:, 1]) + 50), np.int32(min(src_pts[:, 1]) - 50)

# Zobrazení výsledku
plt.figure()
plt.subplot(121)
plt.imshow(img1[y_min:y_max, x_min:x_max], cmap='gray')
plt.axis('equal')

# Aplikace transformace na trojúhelníkovou oblast
triangle_pts = np.float32([dst_pts]).reshape(-1, 1, 2)
transformed_pts = cv2.perspectiveTransform(triangle_pts, M).reshape(triangle_pts.shape[0], -1)

x_max, x_min = np.int32(max(transformed_pts[:, 0]) + 50), np.int32(min(transformed_pts[:, 0]) - 50)
y_max, y_min = np.int32(max(transformed_pts[:, 1]) + 50), np.int32(min(transformed_pts[:, 1]) - 50)

transformed_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

# Vykreslení trojúhelníku na cílové fotografii
cv2.polylines(transformed_img, [np.int32(transformed_pts)], True, (0, 255, 0), 2)

plt.subplot(122)
plt.imshow(transformed_img[y_min:y_max, x_min:x_max], cmap='gray')
plt.axis('equal')
plt.tight_layout()
plt.show()"""


def perform_calculations():
    global points_pos, points_neg, points_cor, points_max, points_track, correlation_area_points_all, \
        current_path_to_photos, gray1, gray2, width, height, keypoints1_sift, descriptors1_sift

    # if 'width' not in globals() or 'height' not in globals() or 'gray1' not in globals():
    # Načtení první a druhé fotografie
    gray1 = load_photo(img_index=0, color_type=photo_type)
    # Naleznete rozměry první fotografie
    height, width = gray1.shape[:2]

    mesh, centers = set_roi()

    if all(item is False for item in do_calculations.values()):
        print("\n\033[33;1m...              Nebyl zvolen žádný typ výpočtu               ...\033[0m"
              "\n\033[32;1m:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\033[0m")
        return

    if scale == 1:
        print("Chcete ručně zadat měřítko?")
        while True:
            if super_speed:
                ans = 'N'
            else:
                ans = askstring("Chcete ručně zadat měřítko?", "Chcete ručně zadat měřítko?\nZadejte Y / N: ")
            # make_scale = input("\tZadejte Y nebo N: ")
            if ans == "Y":
                print("\n\tZvolena možnost 'Y'\n\tMěřítko bude definováno ručně.")
                do_scale(auto_scale=False)
                while True:
                    print("\nChcete proces zopakovat?")
                    rerun = askstring("Chcete proces zopakovat?", "Chcete proces zopakovat?\nZadejte Y / N: ")
                    # rerun = input("\tZadejte Y nebo N: ")
                    if rerun == "Y":
                        do_scale(auto_scale=False)
                    elif rerun == "N":
                        break
                    else:
                        print("\nNeplatná odpověď. Zadejte pouze Y nebo N.")
                del rerun
                break

            elif ans == "N":
                print("\n\tZvolena možnost 'N'\n\tMěřítko bude definováno automaticky.")
                do_scale(img=load_photo(img_index=0, color_type=0), auto_scale=True)
                break
            else:
                print("\nNeplatná odpověď. Zadejte pouze Y nebo N.")
        del ans

    make_angle_correction()

    if (((do_calculations['Do Correlation'] and not calculations_statuses['Correlation']) or
         (do_calculations['Do Rough detection'] and not calculations_statuses['Rough detection']))
            or recalculate['Re Correlation'] or recalculate['Re Rough detection']):

        if not calculations_statuses['Correlation'] or recalculate['Re Correlation']:
            correlation_calculation()

        if ((do_calculations['Do Rough detection'] and not calculations_statuses['Rough detection'])
                or recalculate['Re Rough detection']):
            rough_calculation(mesh, centers)

            del keypoints1_sift, descriptors1_sift

        del mesh, centers

        """for h in range(len(image_files)):  # TODO KONTROLA #########
                    # show_results_graph(show_final_image)  # Vykreslení výsledného grafu fotografie
                    show_results_graph(h)"""

    if (((do_calculations['Do Fine detection'] and not calculations_statuses['Fine detection'])
         or recalculate['Re Fine detection']) and calculations_statuses['Rough detection']):
        """print("\n\033[32m_________________________________________________________________\033[0m"
              "\n\033[32m_________________________________________________________________\033[0m")"""

        fast_fine_calculation(mesh_size=fine_size)
        # fine_calculation_____________(mesh_size=fine_size)
        show_heat_graph(-1, 0, "y", fine_triangle_points_all, fine_mesh_centers_all, scaling=scale,
                        colorbar_label='[mm]', block_graph=block_graphs)  # TODO BLOKACE GRAFU

    if ((do_calculations['Do Point detection'] and not calculations_statuses['Point detection'])
            or recalculate['Re Point detection']):
        print("\n\033[32m-----------------------------------------------------------------\033[0m"
              "\n\033[32m-----------------------------------------------------------------\033[0m")
        point_tracking_calculation(use_correlation=False, interpolate_new_points="H" in data_type,
                                   interpolation_number=16)

    print("\n\033[32;1m.................................................................\033[0m"
          "\n\033[32;1m:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\033[0m")


def match(tem, im, tolerance=0.75, method=cv2.TM_CCOEFF_NORMED):
    h, w = tem.shape[:2]
    r = cv2.matchTemplate(im, tem, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(r)

    if max_val < tolerance:
        print(f"\n\t\033[33;1;21mWARRNING\033[0m\n\t\tOblast nenalezena: korelační koeficien  {max_val:.4f}.")
        return None, None, None, None
    else:
        return np.array(max_loc), h, w, max_val


def do_scale(img=None, auto_scale=True):
    global scale
    if auto_scale:
        if not isinstance(img, np.ndarray):
            print("\n\033[33;1;21mWARRNING\033[0m"
                  "\n\tPro automatické označení měřítka nebyla, nebo byla špatně, zadaná fotografie. "
                  "\n\t\tAutomaticky zvolena první fotografie.")
            img = load_photo(0, photo_type)
        try:
            if scale != 1:
                print("\n\tMěřítko je již definováno, chcete ho i přesto udělat?")
                while True:
                    if super_speed:
                        ans = "Y"
                    else:
                        ans = askstring("Měřítko je již definováno, chcete ho i přesto udělat?",
                                        "Měřítko je již definováno, chcete ho i přesto udělat?\nZadejte Y / N: ")
                        # ans = input("\t\tZadejte Y / N: ")
                    if ans == "Y":
                        print("\n\tZvolena možnost 'Y'\n\tBude definováno nové měřítko.")
                        break
                    elif ans == "N":
                        print(f"\n\tZvolena možnost 'N'\n\tMěřítko nebude znovu definováno.")
                        return
                    else:
                        print("\n Zadejte platnou odpověď.")
        except NameError:
            pass

        scale_paths = os.path.join(folder_measurements, "scale")
        template1 = cv2.imread(scale_paths + r"\12.png", photo_type)[30:-90, 215:-45]
        template2 = cv2.imread(scale_paths + r"\23.png", photo_type)[30:-90, 215:-45]
        top_left1, height1, width1, result1 = match(template1, img, tolerance=0.7)
        top_left2, height2, width2, result2 = match(template2, img, tolerance=0.7)

        if top_left1 is not None and top_left2 is not None:
            dist = np.linalg.norm(top_left1 - top_left2)
            scale = (230 - 120) / dist
            print(f"\n\tMěřítko: {scale:.4f}",
                  f"\n\tCelková vzdálenost: {dist:.3f}\n\t\tRozdíly (x,y): {abs(top_left2 - top_left1)}",
                  f"\n\t\t  Přesnosti: [{result1:.3f}], [{result2:.3f}]")

            if False:
                plt.figure()
                plt.subplot(221)
                plt.imshow(cv2.cvtColor(template1, cv2.COLOR_BGR2RGB))
                plt.subplot(222)
                plt.imshow(cv2.cvtColor(template2, cv2.COLOR_BGR2RGB))
                plt.subplot(223)
                plt.imshow(cv2.cvtColor(
                    img[top_left1[1]:top_left1[1] + height1, top_left1[0]:top_left1[0] + width1], cv2.COLOR_BGR2RGB))
                plt.subplot(224)
                plt.imshow(cv2.cvtColor(
                    img[top_left2[1]:top_left2[1] + height2, top_left2[0]:top_left2[0] + width2], cv2.COLOR_BGR2RGB))
                plt.tight_layout()
                plt.show(block=block_graphs)

            return

        else:
            print("\n\033[33;1;21mWARRNING\033[0m\n\tChyba načtení měřítka.\nZadejte ho ručně.")

    while True:
        dist = mark_points_on_canvas(2)
        if dist:
            break

    print("Zadej vzdáolenost v bodů [mm]")
    while True:
        d_mm = askfloat("Vzdálenost", "Vzdálenost.\nZadejte číslo: ")
        # d_mm = input("\tVzádlenost: ").replace(",", ".")
        try:
            d_mm = abs(np.float64(d_mm))  # pokus o převod na číslo
            break
        except ValueError as ve:
            print(f"Vzdálenost ve špatném formátu, zadajete ji znovu.\n\tPOPIS: {ve}")
            pass

    dist_px = (np.sqrt((dist[1][0] - dist[0][0]) ** 2 + (dist[1][1] - dist[0][1]) ** 2))
    scale = d_mm / dist_px
    print(f"\n\tMěřítko: {scale:.4f}")


def get_index(number):
    tot_im = len(image_files)
    if number > tot_im:
        index = tot_im
    elif number < 0:
        index = tot_im + number
        # index = file.index(file[number])
    else:
        index = number
    return index


def gradient_fill(data_x: list | tuple | np.ndarray = None, data_y: list | tuple | np.ndarray = None,
                  fill_color: str | tuple = None, ax: plt.Axes = None, line: list | plt.Line2D = None,
                  up_alpha: float = None, down_alpha: float = 0.05, z_order: int = None, make_edge: bool = True,
                  gradient_type: str | int = 0, detail_multi: int = 5, maximum_detail=2000):
    if line is None and data_x is None and data_y is None:
        print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatně zadaná data grafu.")
        return
    if not (isinstance(fill_color, (str, tuple)) or fill_color is None):
        print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatně zvolen typ barvy.")
        fill_color = None
    if all((gradient_type != 0, gradient_type != 1, gradient_type != "constant", gradient_type != "smooth")):
        print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatně zvolen typ gradientu.")
        gradient_type = 0

    if all((not isinstance(line, (list, plt.Line2D)), line is not None)):
        print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatně zadaná data linie grafu.")
        return
    elif isinstance(line, list):
        line = line[0]
        if not isinstance(line, plt.Line2D):
            print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatně zadaná linie grafu.")
            return
    if all((not isinstance(ax, plt.Axes), ax is not None)):
        print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatně zadané osy grafu.")
        return
    elif not isinstance(ax, plt.Axes):
        ax = plt.gca()

    if line is None and not (data_x is None and data_y is None):
        max_x, min_x, max_y, min_y = np.max(data_x), np.min(data_x), np.max(data_y), np.min(data_y)
        line = ax.plot(data_x, data_y, color=fill_color)[0]
    else:
        data_x = line.get_xdata()
        data_y = line.get_ydata()
        max_x, min_x, max_y, min_y = np.max(data_x), np.min(data_x), np.max(data_y), np.min(data_y)
    if fill_color is None:
        fill_color = line.get_color()
    if z_order is None:
        z_order = line.get_zorder()

    line_alpha = line.get_alpha()
    up_alpha = 1 if up_alpha is None and line_alpha is None else line_alpha if up_alpha is None else up_alpha
    if up_alpha > 1:
        print("\n\033[37;1;21mWARRNING\033[0m"
              "\033[37m\n\tMaximální průhlednost byla nastavena na hodnotu větší než 1.\033[0m")
    if down_alpha < 0:
        print("\n\033[37;1;21mWARRNING\033[0m"
              "\033[37m\n\tMinimální průhlednost byla nastavena na hodnotu menší než 0.\033[0m")

    detail_multi = max(1, detail_multi)

    data_shape = data_y.shape[0]
    new_length = min(data_shape * detail_multi, maximum_detail)
    rgb = plt.cm.colors.to_rgba(fill_color)[:3]

    if gradient_type == 1 or gradient_type == "smooth":
        quality = min(data_shape * detail_multi, maximum_detail)
        smoothed_data = [(0.5 * np.median(data_y[max(0, i - detail_multi):min(data_shape, i + detail_multi + 1)]) +
                          1.5 * np.mean(data_y[max(0, i - detail_multi):min(data_shape, i + detail_multi + 1)])) / 2
                         for i in range(data_shape)]

        # Vytvoření nového pole s interpolovanými hodnotami
        output_array = np.interp(np.linspace(0, data_shape - 1, new_length), np.arange(data_shape), smoothed_data)

        z = np.ones((new_length, quality, 4), dtype=np.float32)
        z[:, :, :3] = rgb
        for p in range(new_length):
            pos = np.int32(np.ceil((output_array[p] - min_y) / (max_y - min_y) * quality))
            z[p, :pos, -1] = np.linspace(0, 1, pos)
        z = np.transpose(z, axes=(1, 0, 2))
    else:
        z = np.ones((new_length, 1, 4), dtype=np.float32)
        z[:, :, :3] = rgb
        z[:, :, -1] = np.linspace(0, 1, new_length)[:, None]

    min_val = np.min(z[:, :, -1])
    z[:, :, -1] = down_alpha + ((z[:, :, -1] - min_val) / (np.max(z[:, :, -1]) - min_val)) * (up_alpha - down_alpha)
    z[:, :, -1] = np.clip(z[:, :, -1], a_min=0, a_max=1)

    filler = ax.imshow(z, aspect='auto', extent=[min_x, max_x, min_y, max_y], origin='lower', zorder=z_order)

    xy = np.vstack([[min_x, min_y], np.column_stack([data_x, data_y]), [max_x, min_y], [min_x, min_y]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    filler.set_clip_path(clip_path)
    ax.autoscale(True)
    if make_edge:
        # ax.margins(x=(max_x - min_x) * 0.01, y=(max_y - min_y) * 0.01)
        addition_x, addition_y = (max_x - min_x) * 0.025, (max_y - min_y) * 0.025,
        ax.set_xlim(min_x - addition_x, max_x + addition_x)
        ax.set_ylim(min_y - addition_y, max_y + addition_y)


def plot_final_forces(file: str | bytes, correlation_area_coordinates: list | tuple | np.ndarray,
                      show_photos=False, interactive_mode=False, fill_area=False):
    if scale == 1:
        do_scale(load_photo(img_index=0, color_type=0))
    x_data, y_data, photo = load_forces(current_name=file, window_size_average=3)
    if any(data is None for data in (x_data, y_data, photo)):
        print("\n\033[33;1;21mWARRNING\033[0m\n\tNelze vytvořit graf se silami.")
        return

    print("\nData zatížení načtena.")
    distance_correlation = np.linalg.norm(
        np.array(correlation_area_coordinates[0][0][0]) - np.array(correlation_area_coordinates[-1][0][0])) * scale
    distance_measurement = np.linalg.norm(x_data[0] - x_data[-1])
    scale_correction = distance_correlation / distance_measurement
    x_data = x_data * scale_correction

    if interactive_mode:
        plot_width = 7.5
        plot_figs = 2
    else:
        plot_width = 3.5
        plot_figs = 1

    fig, ax = plt.subplots(plot_figs, figsize=(10, plot_width), num="Graph of forces")
    if not isinstance(ax, (np.ndarray, list, tuple)):
        ax = [ax]

    line1 = ax[0].plot(x_data, y_data, color='dodgerblue', zorder=7, label="Force")
    if show_photos:
        ax[0].scatter(x_data[photo], y_data[photo], s=20, marker='o', color='navy', zorder=6, label="Taken photos")
    x_range_start, x_range_end = ax[0].get_xlim()  # rozsah x
    y_range_start, y_range_end = ax[0].get_ylim()  # rozsah y
    # Stejný poměru os
    ax[0].set_aspect(((x_range_end - x_range_start) / (y_range_end - y_range_start)) / 3, adjustable='box')
    ax[0].grid(color="lightgray")
    ax[0].legend(fancybox=True).set_zorder(10)
    ax[0].set_xlabel('Distance [mm]')
    ax[0].set_ylabel('Force [N]')

    if fill_area:
        gradient_fill(ax=ax[0], line=line1, up_alpha=0.6, down_alpha=0.05, fill_color='dodgerblue', make_edge=True,
                      gradient_type=0, detail_multi=250)

    if interactive_mode:
        from matplotlib.widgets import SpanSelector

        # line2, = ax[1].plot([], [])  # , color='royalblue'
        # mark = ax[1].scatter([], [], s=25, marker='o', edgecolor='dodgerblue', facecolor='none', alpha=0.7)
        ax[1].set_xlabel(ax[0].get_xlabel())
        ax[1].set_ylabel(ax[0].get_ylabel())

        def onselect(xmin, xmax):
            ax[1].cla()
            ind_min, ind_max = np.searchsorted(x_data, (xmin, xmax))
            ind_max = min(len(x_data) - 1, ind_max)
            ind_min = max(ind_min, 0)

            region_x = x_data[ind_min:ind_max]
            region_y = y_data[ind_min:ind_max]

            if show_photos:
                p = photo[(photo >= ind_min) & (photo <= ind_max)]
                # mark.set_offsets((x_data[p], y_data[p])) if len(p) > 0 else None
                ax[1].scatter(x_data[p], y_data[p], s=25, marker='o', edgecolor='dodgerblue', facecolor='none',
                              alpha=0.7)

            if len(region_x) >= 2:
                mar_x, mar_y = (region_x[-1] - region_x[0]) * 0.025, (region_y.max() - region_y.min()) * 0.025
                # line2.set_data(region_x, region_y)
                ax[1].plot(region_x, region_y)
                ax[1].set_xlim(region_x[0] - mar_x, region_x[-1] + mar_x)
                ax[1].set_ylim(region_y.min() - mar_y, region_y.max() + mar_y)
                fig.canvas.draw_idle()

        _ = SpanSelector(  # span
            ax[0],
            onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True
        )
        # Set useblit=True on most backends for enhanced performance.

    plt.tight_layout()
    """x_mar, y_mar = (np.max(x_data) - np.min(x_data)) * 0.05, (np.max(y_data) - np.min(y_data)) * 0.05
    for axis in ax:
        axis.margins(x=x_mar, y=x_mar)
        axis.autoscale(True)"""
    plt.show()


def show_results_graph(image_number, img_color=0):
    index = get_index(image_number)

    print("\n  -  Vykreslení výsledků: Fotografie - {}: {}".format(index + 1, image_files[index]))

    image = load_photo(img_index=index, color_type=img_color)

    h, w = image.shape[:2]

    fig, ax = plt.subplots(figsize=(np.int8(0.0017 * w), np.int8(0.0017 * h)), num="Final result graph")
    plt.title('Triangle elements - Image {}: {}'.format(index + 1, image_files[index]), wrap=True)

    ax.imshow(image, cmap='gray')
    fig.set_facecolor('none')
    ax.set_facecolor('none')

    try:
        triangles = triangle_vertices_all[index]
        new_center = triangle_centers_all[index]
        tri_index = triangle_indexes_all[index]
        triangle_points = triangle_points_all[index]
        wrong_triangle_coordinates = triangle_points[wrong_points_indexes_all[index]]

        [ax.add_patch(Polygon(triangle, closed=True, facecolor='darkorange', edgecolor='none', alpha=0.45)) for
         triangle in wrong_triangle_coordinates]

        ax.triplot(triangles[:, 0], triangles[:, 1], tri_index, color='green')

        ax.scatter(new_center[:, 0], new_center[:, 1], s=8, marker='o')
    except NameError:
        pass

    try:
        upper_area_cor = correlation_area_points_all[index]

        # Vykreslení obdélníků
        for area in upper_area_cor:
            rec_w, rec_h = area[1, 0] - area[0, 0], area[1, 1] - area[0, 1]
            ax.add_patch(Rectangle((area[0]), rec_w, rec_h, edgecolor='firebrick', lw=1.5, facecolor='none'))
            ax.add_patch(Rectangle((area[0]), rec_w, rec_h, facecolor='red', alpha=0.1))
    except NameError:
        pass

    plt.axis('off')
    # plt.subplots_adjust(right=0.99, left=0.05, top=0.96, bottom=0.045)
    try:
        plt.tight_layout()
        ax.set_aspect('equal', adjustable='box')
        ax.autoscale(True)

        plt.pause(0.5)
        plt.show(block=block_graphs)
        plt.pause(2)
    except UserWarning:
        plt.close("Final result graph")


def plot_point_path(image_number, img_color=0, plot_correlation_paths=False, plot_tracked_paths=False,
                    plot_rough_paths=False, plot_fine_paths=False, show_menu=True, show_cor=False, show_areas=False,
                    indexes_pt='all', indexes_cor='all', index_pt='all', index_cor='all', text_size: float | int = 9):
    index = get_index(image_number)

    print("\n  -  Vykreslení průběhu bodů: Fotografie - {}: {}".format(index + 1, image_files[index]))

    image = load_photo(img_index=index, color_type=img_color)
    h, w = image.shape[:2]

    def update(name):
        [obj.set_visible(not obj.get_visible()) for obj in objets[name]]
        if name == "Souřadnice 1" or name == "Souřadnice 2":
            if check.get_status()[-1] or check.get_status()[-2]:
                [obj.set_visible(True) for obj in arrow]
            else:
                [obj.set_visible(False) for obj in arrow]
            [obj.figure.canvas.draw_idle() for obj in arrow]
        [obj.figure.canvas.draw_idle() for obj in objets[name]]

    try:
        # correlation_points = correlation_area_points_all  # correlation areas
        # tracked_points = tracked_points_all  # tracked points
        # rough_points = triangle_centers_all  # rough element
        # fine_points = fine_mesh_centers_all  # fine element

        index_ = []
        for ind_1, ind_2, coordinates in zip((index_cor, index_pt), (indexes_cor, indexes_pt),
                                             (correlation_area_points_all, tracked_points_all)):
            length_1, length_2 = len(coordinates), len(coordinates[0])
            for ind, length in zip((ind_1, ind_2), (length_1, length_2)):
                if isinstance(ind, str) and ind == 'all':
                    ind = list(range(length))
                elif isinstance(ind, (list, tuple, np.ndarray)):
                    if np.max(ind) > length - 1 or np.min(ind) < -length:
                        return
                index_.append(ind)
        [index_cor, indexes_cor, index_pt, indexes_pt] = index_

        areas = [np.array([correlation_area_points_all[i][j] for i in index_cor]) for j in indexes_cor]
        points = [np.array([tracked_points_all[i][j] for i in index_pt]) for j in indexes_pt]
        # points = [np.array([point[j] for point in tracked_points_all]) for j in range(len(tracked_points_all[0]))]

        areas = [make_angle_correction(points_to_warp=a).reshape(-1, 2, 2) for a in areas]
        points = [make_angle_correction(points_to_warp=p) for p in points]
        image = make_angle_correction(image_to_warp=image)

        fig, ax = plt.subplots(figsize=(np.int8(0.002 * w), np.int8(0.0017 * h)), num="Points paths graph")
        plt.title('Points paths - Image {}: {}'.format(index + 1, image_files[index]), wrap=True, pad=12)

        box_width = 0.2 if show_menu else 0

        fig.subplots_adjust(right=1 - box_width - 0.02, left=0.02, top=0.9, bottom=0.02, wspace=0, hspace=0)

        ax.imshow(image, cmap='gray')
        fig.set_facecolor('none')
        ax.set_facecolor('none')

        arrow_length = min(image.shape[0] * 0.2, image.shape[1] * 0.2)
        # Určete body polygonu pro šipku
        ar_points = np.int16([
            (0, 0), (arrow_length, 0), (arrow_length * 0.6, arrow_length * 0.3),
            (arrow_length * 0.6, arrow_length * 0.15), (arrow_length * 0.15, arrow_length * 0.15)])
        ar_points = np.vstack((ar_points, (ar_points[:, [1, 0]])[:0:-1]))

        arrow = [ax.add_patch(Polygon(ar_points, closed=True, facecolor='black', edgecolor='white', lw=1.5, alpha=0.6))]
        # Vykreslete šipku jako polygon
        for i in ((1.1, 0.2, 'x'), (0.1, 1.25, 'y'), (0.85, 1, 'r')):
            arrow.append(
                ax.text(arrow_length * i[0], arrow_length * i[1], i[2], fontweight='bold', color='white',
                        fontsize=max(arrow_length // 65, 8),
                        bbox=dict(facecolor='black', edgecolor='white', boxstyle='round', pad=0.2, alpha=0.6)))
        arrow.append(ax.add_patch(FancyArrowPatch(
            (arrow_length * 0.2, arrow_length * 0.95), (arrow_length * 0.95, arrow_length * 0.2),
            facecolor='black', edgecolor='white', mutation_scale=0.075, alpha=0.6, connectionstyle="arc3,rad=0.45",
            linewidth=1.5, arrowstyle=f"Simple, head_length={arrow_length * 0.35}, head_width={arrow_length * 0.4},"
                                      f"tail_width={arrow_length * 0.12}")))

        texts1, areas1 = [], []
        if plot_correlation_paths:
            for i, area in enumerate(areas):  # Vykreslení korelačních oblastí
                area1 = Rectangle((area[index][0]), area[index][1, 0] - area[index][0, 0],
                                  area[index][1, 1] - area[index][0, 1], facecolor='firebrick', alpha=0.5, zorder=0)
                areas1.append(area1)
                ax.add_patch(area1)

                centers = np.array([[np.mean(rect[:, 0]), np.mean(rect[:, 1])] for rect in area])
                ax.plot(centers[:, 0], centers[:, 1], linestyle=(0, (5, 5)), marker='o', markersize=3.5,
                        c='#EA494F', label=f'Correlation area {i + 1}.')
                ax.scatter(centers[index, 0], centers[index, 1], s=20, marker='o', color='#731331', zorder=2)
                ax.text(
                    centers[index, 0] + 20, centers[index, 1] - 20, f"{i + 1}", fontsize=text_size, ha='left',
                    va='bottom', color='#ED9AB0', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='#320015')])
                text1 = ax.text(centers[index, 0] + 40, centers[index, 1] + 20, f"{np.int32(np.round(centers[index]))}",
                                fontsize=text_size - 2, ha='left', va='top', color='#ED9AB0', fontweight='bold',
                                style='italic', path_effects=[pe.withStroke(linewidth=2, foreground='#320015')])
                texts1.append(text1)

        texts2, areas2 = [], []
        if plot_tracked_paths:
            for i, point in enumerate(points):  # Vykreslení hledaných bodů  ([points[i][index_pt] for i in indexes_pt])
                area2 = plt.Circle(point[index], 40, edgecolor='none', facecolor='#34B4F4', alpha=0.5, zorder=0)
                areas2.append(area2)
                ax.add_artist(area2)

                ax.plot(point[:, 0], point[:, 1], linestyle=(0, (5, 5)), marker='o', markersize=3.5, c='#0679C3',
                        label=f'Tracked point {i + 1}.')
                ax.scatter(point[index, 0], point[index, 1], s=20, marker='o', color='#1D3485', zorder=2)
                ax.text(point[index, 0] + 20, point[index, 1] - 20, f"{i + 1}", fontsize=text_size, fontweight='bold',
                        ha='left',
                        va='bottom', color='#99E6FF', path_effects=[pe.withStroke(linewidth=2, foreground='#051C2C')])
                text2 = ax.text(point[index, 0] + 40, point[index, 1] + 20, f"{point[index]}",
                                fontsize=text_size - 2, ha='left', va='top', color='#99E6FF', fontweight='bold',
                                style='italic', path_effects=[pe.withStroke(linewidth=2, foreground='#051C2C')])
                texts2.append(text2)
    except NameError:
        print("\nChyba u vykreslení označených bodů.")
        return

    objets = {'Oblast 1': areas1, 'Oblast 2': areas2, 'Souřadnice 1': texts1, 'Souřadnice 2': texts2}
    # left, bottom, width, height
    rax = fig.add_axes((fig.subplotpars.right + 0.01, fig.subplotpars.top - 0.25, box_width * 0.75, 0.25))
    check = CheckButtons(ax=rax, labels=objets.keys(), actives=[show_areas, show_areas, show_cor, show_cor],
                         useblit=True, frame_props={'linewidth': [2, ], 'sizes': [100, ]},
                         label_props={'fontsize': [20, ], 'fontweight': ['bold', ]})
    rax.set_visible(show_menu)
    # Nastavení velikosti a barvy textu v tlačítcích
    [(label.set_fontsize(9), label.set_color('black'), label.set_fontweight('bold')) for label in check.labels]
    rax.add_patch(FancyBboxPatch((0, 0), 1, 1, edgecolor='gray', facecolor='#CCCCCC', alpha=0.8, zorder=0,
                                 boxstyle="round,pad=0,rounding_size=0.08", mutation_aspect=1.1, ))

    check.on_clicked(update)
    [update(key) for key in objets.keys()]

    h, n = ax.get_legend_handles_labels()
    leg_ax = rax if show_menu else ax
    leg = leg_ax.legend(h, n, borderaxespad=0.1, title='Legenda', frameon=True, title_fontsize=11, edgecolor='#AEAEAE',
                        fontsize=9, framealpha=1 if show_menu else 0.95, fancybox=True, shadow=show_menu, ncols=1)
    (leg.set_bbox_to_anchor((0.037, 0)), leg.set_loc('upper left')) if show_menu else None

    ax.axis('off')
    rax.axis('off')
    # plt.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    rax.set_aspect('equal', adjustable='box')
    ax.autoscale(True)
    rax.autoscale(True)

    plt.pause(0.5)
    plt.show(block=block_graphs or show_menu)
    plt.pause(2)


def plot_marked_points(image_number=0, img_color=0, make_title=False, indexes='all', save_plot=False,
                       plot_name="marked_points", plot_format="pdf", save_dpi=300, show_menu=True, show_cor=False,
                       show_arrows=False, text_size: float | int = 9, show_marked_points=True):
    index = get_index(image_number)

    print("\n  -  Vykreslení označených bodů: Fotografie - {}: {}".format(index + 1, image_files[index]))

    image = load_photo(img_index=index, color_type=img_color)
    h, w = image.shape[:2]

    def update(name):
        [obj.set_visible(not obj.get_visible()) for obj in objets[name]]
        [obj.figure.canvas.draw_idle() for obj in objets[name]]

    try:
        if save_plot:
            show_menu = False

        fig, ax = plt.subplots(figsize=(np.int8(0.002 * w), np.int8(0.0017 * h)), num="Marked points graph")
        if make_title:
            plt.title('Marked points - Image {}: {}'.format(index + 1, image_files[index]), wrap=True, pad=12)

        box_width = 0.2 if show_menu else 0
        fig.subplots_adjust(right=1 - box_width - 0.02, left=0.02, top=0.9, bottom=0.02, wspace=0, hspace=0)

        ax.imshow(image, cmap='gray')
        fig.set_facecolor('none')
        ax.set_facecolor('none')

        arrow_length = min(image.shape[0] * 0.2, image.shape[1] * 0.2)
        # Určete body polygonu pro šipku
        ar_points = np.int16([
            (0, 0), (arrow_length, 0), (arrow_length * 0.6, arrow_length * 0.3),
            (arrow_length * 0.6, arrow_length * 0.15), (arrow_length * 0.15, arrow_length * 0.15)])
        ar_points = np.vstack((ar_points, (ar_points[:, [1, 0]])[:0:-1]))
        loc = [ax.add_patch(Polygon(ar_points, closed=True, facecolor='#920422', edgecolor='white', lw=1.5, alpha=0.6))]
        for i in ((1.1, 0.2, 'x'), (0.1, 1.25, 'y'), (0.85, 1, 'r')):  # (0.85, 0.45, 'x'), (0.25, 1.1, 'y')
            loc.append(
                ax.text(arrow_length * i[0], arrow_length * i[1], i[2], fontweight='bold', color='white',
                        fontsize=max(arrow_length // 65, 8),
                        bbox=dict(facecolor='#920422', edgecolor='white', boxstyle='round', pad=0.2, alpha=0.6)))
        loc.append(ax.add_patch(FancyArrowPatch(
            (arrow_length * 0.2, arrow_length * 0.95), (arrow_length * 0.95, arrow_length * 0.2),
            facecolor='#920422', edgecolor='white', mutation_scale=0.075, alpha=0.6, connectionstyle="arc3,rad=0.45",
            linewidth=1.5, arrowstyle=f"Simple, head_length={arrow_length * 0.35}, head_width={arrow_length * 0.4},"
                                      f"tail_width={arrow_length * 0.12}")))

        if index == 0 and show_marked_points and "points_track" not in globals():
            try:
                set_roi(just_load=True)
            except (Exception, MyException) as e:
                print(f"\nNepovedlo se načíst označené oblasti: {e}")
        if index == 0 and show_marked_points and "points_track" in globals():
            length = len(points_track)
            if isinstance(indexes, str) and indexes == 'all':
                indexes = list(range(length))
            elif isinstance(indexes, (list, tuple, np.ndarray)):
                if np.max(indexes) > length - 1 or np.min(indexes) < -length:
                    return

            [(ax.add_artist(plt.Circle(point[0], 35, edgecolor='none', facecolor='#FF96BA', alpha=0.3)),
              ax.scatter(point[0][0], point[0][1], s=35, marker='X', edgecolor='white', facecolor='#920422'),
              ax.text(point[0][0] + 20, point[0][1] - 20, f"{n + 1}", fontsize=text_size, fontweight='bold', ha='left',
                      va='bottom', color='#920422', path_effects=[pe.withStroke(linewidth=2, foreground='white')]),
              loc.append(ax.text(point[0][0] + 40, point[0][1] + 20, f"{point[0]}", fontsize=text_size - 2, ha='left',
                                 va='top', color='#ED9AB0', fontweight='bold', style='italic',
                                 path_effects=[pe.withStroke(linewidth=2, foreground='#320015')])))
             for n, point in enumerate([points_track[i] for i in indexes])]

        else:
            length = len(tracked_points_all[0])
            if isinstance(indexes, str) and indexes == 'all':
                indexes = list(range(length))
            elif isinstance(indexes, (list, tuple, np.ndarray)):
                if np.max(indexes) > length - 1 or np.min(indexes) < -length:
                    return

            points = np.array([point for point in tracked_points_all[index]])[indexes]
            [(ax.add_artist(plt.Circle(point, 35, edgecolor='none', facecolor='#FF96BA', alpha=0.3)),
              ax.scatter(point[0], point[1], s=35, marker='X', edgecolor='white', facecolor='#920422'),
              ax.text(point[0] + 20, point[1] - 20, f"{n + 1}", fontsize=text_size, fontweight='bold', ha='left',
                      va='bottom', color='#920422', path_effects=[pe.withStroke(linewidth=2, foreground='white')]),
              loc.append(ax.text(point[0] + 40, point[1] + 20, f"{point}", fontsize=text_size - 2, ha='left',
                                 va='top', color='#ED9AB0', fontweight='bold', style='italic',
                                 path_effects=[pe.withStroke(linewidth=2, foreground='#320015')])))
             for n, point in enumerate(points)]
    except NameError:
        print("\nChyba u vykreslení označených bodů.")
        return

    objets = {'Coordinates': loc}
    # left, bottom, width, height
    rax = fig.add_axes(((fig.subplotpars.right + 0.01), fig.subplotpars.top - 0.25, box_width * 0.75, 0.25))
    check = CheckButtons(ax=rax, labels=objets.keys(), actives=[show_cor], useblit=True,
                         frame_props={'linewidth': [2, ], 'sizes': [100, ]},
                         label_props={'fontsize': [20, ], 'fontweight': ['bold', ]})
    rax.set_visible(show_menu)
    # Nastavení velikosti a barvy textu v tlačítcích
    [(label.set_fontsize(9), label.set_color('black'), label.set_fontweight('bold')) for label in check.labels]
    rax.add_patch(FancyBboxPatch((0, 0), 1, 1, edgecolor='gray', facecolor='#CCCCCC', alpha=0.8, zorder=0,
                                 boxstyle="round,pad=0,rounding_size=0.08", mutation_aspect=1.1, ))

    check.on_clicked(update)
    [update(key) for key in objets.keys()]
    ax.axis('off')
    rax.axis('off')
    # plt.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    rax.set_aspect('equal', adjustable='box')
    ax.autoscale(True)
    rax.autoscale(True)

    if show_arrows:
        [obj.set_visible(not obj.get_visible()) for obj in objets['Coordinates'][:5]]
        [obj.figure.canvas.draw_idle() for obj in objets['Coordinates'][:5]]

    if save_plot:
        plot_format = plot_format.replace(".", "")
        if plot_format not in ('jpg', 'jpeg', 'JPG', 'eps', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz',
                               'tif', 'tiff', 'webp'):
            print(f"\nNepodporavaný formát [ *.{plot_format} ], automaticky změněno na [ *.pdf ]")
            plot_format = 'pdf'
        if plot_name is None:
            plot_name = "marked_points"
        plot_name = plot_name.replace(".", "_") + "." + plot_format
        plt.savefig(os.path.join(current_folder_path, plot_name), format=plot_format, dpi=save_dpi, bbox_inches='tight')
        plt.close("Marked points graph")
    else:
        plt.pause(0.5)
        plt.show(block=block_graphs or show_menu)
        plt.pause(2)


def show_heat_graph(image_index_shift, image_index_background, axes, coordinates, centers=None, heat_values=None,
                    line_values=None, scaling: float = 1, heat_graph_title=None, line_graph_title=None,
                    graph_title=None, make_line_graph=False, figure_rgba=(1, 1, 1, 1), main_ax_rgba=(0, 0, 0, 0),
                    line_graph_main_ax_rgba=(0, 0, 0, 0), line_graph_sub_ax_rgba=(1, 1, 1, 1),
                    fill_between=False, min_val=None, max_val=None, colorbar_spacing=9, colorbar_style='jet',
                    colorbar_label=None, get_graph=False, saved_graph_name=None, save_graph=False,
                    save_graph_separately=False, graph_format="pdf", save_dpi=300, use_latex=False,
                    show_correlation_areas=False, correlation_values=None, image_color_type=0, block_graph=True,
                    make_masked_image=False):
    print("\nVytvoření grafu posunů.")

    global correlation_area_points_all

    tot_im = len(image_files)

    shift_index = get_index(image_index_shift)
    background_index = get_index(image_index_background)

    if scale == 1:
        do_scale(load_photo(img_index=0, color_type=0))

    axes = str(axes).lower()
    if axes == "y" or axes == '1':
        direction = [1]
    elif axes == "x" or axes == '0':
        direction = [0]
    elif axes == "both" or axes == '3':
        direction = [0, 1]
    elif axes == "tot" or axes == '2':
        direction = [2]
    elif axes == "tot2" or axes == '4':
        direction = [2]
    elif axes == "none" or axes == '5':
        direction = [None]
        make_line_graph = True
    else:
        print("\n\033[31;1;21mERROR\033[0m\nŠpatně zvolený směr posunu.\n\t - Graf nevykreslen.")
        return

    if (line_values is not None or make_line_graph) and direction != [None]:
        direction = [None] + direction

    axis = ["x", "y", "tot"]
    figures = len(direction)

    img = cv2.cvtColor(load_photo(img_index=background_index, color_type=image_color_type), cv2.COLOR_BGR2RGBA)

    if make_masked_image:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # Vykreslení oblastí na masce
        [cv2.fillPoly(mask, [np.int32(np.round(polygon))], 255) for polygon in triangle_points_all[background_index]]

        if show_correlation_areas:
            [cv2.rectangle(mask, np.int32(np.round(rectangle[0])), np.int32(np.round(rectangle[1])), 255, -1)
             for rectangle in correlation_area_points_all[background_index]]

    ratio = (img.shape[1] / img.shape[0])
    if save_graph:
        fig_size = 5
    elif figures == 3:
        fig_size = 16 / (figures * ratio)
        ratio = ratio + 1.5
    elif figures * 5 * ratio > 16:
        fig_size = 16 / (figures * ratio)
    else:
        fig_size = 5

    # LaTeX setup
    if ((save_graph or save_graph_separately) and graph_format in ("pgf", "eps", "ps") and use_latex) or use_latex:
        plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'  # load times roman font
        plt.rcParams['font.family'] = 'serif'  # use serif font as default
        plt.rcParams['text.usetex'] = True  # enable LaTeX rendering globally

    if figures == 3:
        save_graph = save_graph_separately = False

    fig, axs = plt.subplots(1, figures, figsize=(figures * fig_size * ratio, fig_size), num="Final displacement graph")
    if not isinstance(axs, (np.ndarray, list, tuple)):
        axs = [axs]
    else:
        if graph_title is None:
            graph_title = "Displacements"
        plt.suptitle(graph_title, y=0.99, fontweight='bold', wrap=True)

    cor_values = None
    """import seaborn as sns
    sns.set_theme()"""  # TODO SEABORN styl grafu

    for ax, dic in zip(axs, direction):
        if (line_values is not None or make_line_graph) and ax == axs[0]:
            ax.axis('off')

            if line_values is None:
                if 'correlation_area_points_all' not in globals():
                    print("\n\033[33;1;21mWARRNING\033[0m\n\tNelze vytvořit graf se silami.")
                    continue

                x_data, y_data, photos = load_forces(current_name=current_image_folder, window_size_average=1)
                if any(data is None for data in (x_data, y_data, photos)):
                    print("\n\033[33;1;21mWARRNING\033[0m\n\tNelze vytvořit graf se silami.")
                    continue
                x_data = x_data * ((np.linalg.norm(
                    np.array(correlation_area_points_all[-1][0][0]) - np.array(correlation_area_points_all[0][0][0]))
                                    * scaling) / np.linalg.norm(x_data[-1] - x_data[0]))
                photos = photos[start:end]  # TODO mít nebo tam nemusí být start?
            else:
                x_data, y_data, photos = line_values

            photos_taken = len(photos)

            if line_graph_title is None:
                line_graph_title = 'Total force\n'
            ax.set_title(line_graph_title, pad=10, wrap=True)

            # Časové úseky pořízení fotografií
            top_pad = 0.86
            try:
                time_period, time_stamps = None, []
                if not tot_im >= 2:
                    raise MyException("Minimální počet fotek pro tvorbu časových razítek je 2.")

                time_stamps, time_period = get_photos_time_stamps()

                if time_period is not None or time_stamps:
                    top_pad = 0.75
                else:
                    raise MyException("\033[33mNebylo možné získat čas ze zadávací sekvence nebo "
                                      "času vytvoření fotografií\033[0m")

                if time_period is None:
                    if time_stamps:
                        if len(photos_times) >= 3:
                            time_period = np.median(time_stamps[1:-1])
                            time_stamps[1:-1] = np.median(time_stamps[1:-1])
                        else:
                            time_period = time_stamps[-1]
                    else:
                        time_stamps = []
                        for i in (0, 1):
                            image_path = load_photo(img_index=i, give_path=True)
                            if os.path.exists(image_path):
                                time_stamps.append(os.path.getmtime(image_path))
                        if len(time_stamps) == 2:
                            time_stamps = np.int16(abs(time_stamps[0] - time_stamps[1]))

                if photos[-1] + 1 == len(x_data) and tot_im >= 3:
                    if 'time_stamps' in locals() and isinstance(time_stamps, list) and len(time_stamps) >= 3:
                        last_time = np.int16(abs(time_stamps[-2] - time_stamps[-1]))
                        del time_stamps
                    else:
                        last_time = []
                        for i in (-1, -2):
                            image_path = load_photo(img_index=i, give_path=True)
                            # image_path = os.path.join(current_folder_path, source_image_type[0], image_files[i])
                            if os.path.exists(image_path):
                                last_time.append(os.path.getmtime(image_path))
                        if len(last_time) == 2:
                            last_time = np.int16(abs(last_time[0] - last_time[1]))
                        else:
                            last_time = None
                else:
                    last_time = None

                if last_time == time_period:
                    last_time = None

            except (Exception, MyException) as e:
                print("\n\033[33;1;21mWARRNING\033[0m\n\t - "
                      f"Chyba načtení časového nastavení měření ze složky: [{current_image_folder}.txt]\n\tPOPIS: {e}")
                time_period = None

            if direction == [None]:
                right_pad = 0.88
            else:
                right_pad = 0.93

            inner_axes = ax.inset_axes([0.1, 0.12, right_pad, top_pad])  # left, down, right, up : 0.1, 0.12, 0.88, 0.86
            inner_axes.scatter(x_data[photos[:shift_index]], y_data[photos[:shift_index]], s=20, marker='o', zorder=10,
                               edgecolor='dodgerblue', facecolor="white", label="Taken photos")
            inner_axes.scatter(x_data[photos[shift_index:]], y_data[photos[shift_index:]], s=20, marker='o', zorder=10,
                               edgecolor=ctu_color['CTU_skyblue'], facecolor="white")
            photo = photos[shift_index]
            inner_axes.scatter(x_data[photo], y_data[photo], s=21, marker='o', zorder=11,
                               edgecolor='dodgerblue', facecolor="navy", label="Displayed photo")
            inner_axes.plot(x_data[:photo], y_data[:photo], c='dodgerblue', zorder=9, label='Forces')
            inner_axes.plot(x_data[photo:], y_data[photo:], c=ctu_color['CTU_skyblue'], zorder=8, alpha=0.75, lw=0.75)
            if fill_between:
                inner_axes.fill_between(x_data[photos[:shift_index]], y_data[photos[:shift_index]], color='dodgerblue',
                                        alpha=0.25, label=None, zorder=7)
                inner_axes.fill_between(x_data[photos[shift_index:]], y_data[photos[shift_index:]], color='skyblue',
                                        alpha=0.25, label=None, zorder=7)

            # Získání rozsahů os x a y na začátku
            # x_range_start, x_range_end = np.min(x_data), np.max(x_data)
            # y_range_start, y_range_end = np.min(y_data), np.max(y_data)

            # Nastavení stejného poměru os x a y při přibližování
            # inner_axes.set_aspect(((x_range_end - x_range_start) / (y_range_end - y_range_start)) / 3)

            # x_ticks = np.sort(np.append(x_ticks, 0))
            # tick_labels = ["" if item == "0,00" else item for item in tick_labels]
            for i in ((inner_axes.set_xticks, x_data, 9, inner_axes.set_xticklabels, inner_axes.get_xticks),
                      (inner_axes.set_yticks, y_data, 7, inner_axes.set_yticklabels, inner_axes.get_yticks)):
                i[0](np.linspace(0, np.max(i[1]), i[2]))  # np.int32(np.min(i[1])) TODO počátek ?????
                i[3]([f"{label:.2f}".replace('.', ',') for label in sorted(i[4]().tolist())])  # Přepsání hodnot na ose

            inner_axes.grid(color="lightgray")
            for i in ((inner_axes.axhline, None), (inner_axes.axvline, None)):
                i[0](0, color='black', linewidth=1.4, linestyle='-', alpha=0.3, zorder=3, label=i[1])  # "Zero state"

            inner_axes.set_xlabel('Distance [mm]', labelpad=5)
            inner_axes.set_ylabel('Force [N]', labelpad=3)
            inner_axes.legend(fontsize=9, fancybox=True)

            # Vytvoření druhého grafu s druhými osami (skkryté) pro fungování posunu a přiblížení
            for i in (inner_axes.twinx(), inner_axes.twiny()):
                i.axis('off')
            # sec_inner_axes.set_yticks([])

            if time_period is not None:
                # Vytvoření druhé x-osy s vlastním měřítkem
                # sc = (x_data[-1] - x_data[0]) / (photos_taken - 1)
                sec_ax = inner_axes.secondary_xaxis('top')  # , functions=(lambda x: (x - x_data[0]) / sc, lambda x: x))
                sec_ax.set_xlabel(f'Time [s]', labelpad=8)

                index = np.linspace(0, photos_taken - 1, min(photos_taken, 10), dtype=np.int32)
                sec_ax.xaxis.set_major_locator(plt.FixedLocator(x_data[photos][index]))
                sec_ax.xaxis.set_minor_locator(plt.FixedLocator(np.delete(x_data[photos], index)))
                # sec_ax.set_xticks(x_data[photos])

                if last_time is not None:
                    x_ticks = np.arange(0, photos_taken - 1, 1, dtype=np.int16) * time_period
                    x_ticks = np.append(x_ticks, x_ticks[-1] + last_time)
                    ax.text(0, -0.0075, f'(photo intervals until the last one: {time_period} s)', fontsize=8,
                            ha='left', va='bottom')
                else:
                    x_ticks = np.arange(0, photos_taken, 1, dtype=np.int16) * time_period
                    ax.text(0, -0.0075, f'(photo intervals: {time_period} s)', fontsize=8, ha='left', va='bottom')

                x_ticks = [f"{label:.0f}" for label in np.sort(x_ticks[index])]
                """tick_labels = [""] * photos_taken
                for i in index:  # min(320 // photos_taken, 10)
                    tick_labels[i] = x_ticks[i]"""
                sec_ax.set_xticklabels(x_ticks, fontsize=9)
                # Nastavení stylu čáry pro ticky                                              # 'inout', 'in', 'out'
                sec_ax.tick_params(which='major', axis='x', length=5, width=0.8, color='gray', direction='out')
                sec_ax.tick_params(which='minor', axis='x', length=3.5, width=0.8, color=ctu_color['CTU_gray'],
                                   direction='out')
                sec_ax.spines['top'].set_linewidth(0)

            # Nastavení stylu čar pro horní hranu grafu (spiny)
            for i in (('top', 0.7), ('bottom', 0.7), ('left', 0.7), ('right', 0.7)):
                inner_axes.spines[i[0]].set_color('gray')
                inner_axes.spines[i[0]].set_linewidth(i[1])
                # inner_axes.spines[i].set_linestyle('-')

            for i in ("x", "y"):
                inner_axes.tick_params(which='major', axis=i, length=5, width=0.8, color='gray', direction='out')

            h, n = inner_axes.get_legend_handles_labels()
            inner_axes.legend(h, n, fontsize=9, bbox_to_anchor=[0.5, -0.31],
                              **dict(ncol=len(h), loc="lower center", frameon=True, fancybox=True))
            """h, l = inner_axes.get_legend_handles_labels()
            s = max(round(len(h) / 2), 4)
            kw = dict(ncol=s, loc="lower center", frameon=True)
            leg1 = inner_axes.legend(h[:s], l[:s], fontsize=9, bbox_to_anchor=[0.5, -0.305], **kw)
            leg2 = inner_axes.legend(h[s:], l[s:], bbox_to_anchor=(0, -0.2, 1, 0.2), **kw)  # (x, y, width, height)
            inner_axes.add_artist(leg1)"""

            inner_axes.set_facecolor(line_graph_sub_ax_rgba)
            ax.set_facecolor(line_graph_main_ax_rgba)

            # inner_axes.set_xlim(np.min(x), np.max(x))
            # inner_axes.set_ylim(np.min(y1), np.max(y1))
            # inner_axes.axis('equal')
        else:
            if correlation_values is not None:
                if correlation_values.ndim > 1:
                    if axes == "tot" or axes == '2' or axes == "tot2" or axes == '4':
                        print("\n\033[33;1;21mWARRNING\033[0m\n\t - Chybně zadaná data s typem grafu.")
                        return
                    cor_values = correlation_values[:, dic]
                else:
                    cor_values = correlation_values

            if heat_values is not None:
                if heat_values.ndim > 1:
                    if axes == "tot" or axes == '2' or axes == "tot2" or axes == '4':
                        print("\n\033[33;1;21mWARRNING\033[0m\n\t - Chybně zadaná data s typem grafu.")
                        return
                    subregion_values = heat_values[:, dic]
                else:
                    subregion_values = heat_values

                if dic == 1:
                    try:
                        subregion_values += x_data[0]
                    except (NameError, ValueError):
                        x_data, _, _ = load_forces(current_name=current_image_folder, window_size_average=1)
                        if x_data is not None:
                            x_data = x_data * ((np.linalg.norm(np.array(correlation_area_points_all[-1][0][0]) -
                                                               np.array(correlation_area_points_all[0][0][0]))
                                                * scaling) / np.linalg.norm(x_data[-1] - x_data[0]))
                            subregion_values += x_data[0]

                try:
                    if min_val is None:
                        try:
                            min_value = min(np.min(subregion_values), np.min(cor_values))
                        except TypeError:
                            min_value = np.min(subregion_values)
                    elif len(min_val) > 1:
                        min_value = min_val[dic]
                except TypeError:
                    min_value = min_val

                try:
                    if max_val is None:
                        try:
                            max_value = max(np.max(subregion_values), np.max(cor_values))
                        except TypeError:
                            max_value = np.max(subregion_values)
                    elif len(max_val) > 1:
                        max_value = max_val[dic]
                except TypeError:
                    max_value = max_val

            elif centers is not None:
                if show_correlation_areas:
                    if axes == "tot" or axes == '2':
                        cor_values = (
                                np.linalg.norm(np.array(correlation_area_points_all[shift_index][:][0][0]) -
                                               np.array(correlation_area_points_all[0][:][0][0]), axis=1) * scaling)
                    elif axes == "tot2" or axes == '4':
                        cor_values = (np.array(correlation_area_points_all[shift_index][:][0][0]) -
                                      np.array(correlation_area_points_all[0][:][0][0])).reshape(-1, 2)
                        cor_values = (cor_values[:, 0] + cor_values[:, 1]) * scaling
                    elif dic == 0:
                        cor_values = (correlation_area_points_all[shift_index][:][0][0, dic] -
                                      correlation_area_points_all[0][:][0][0, dic]) * scaling
                    elif dic == 1:
                        cor_values = (correlation_area_points_all[shift_index][:][0][0, dic] -
                                      correlation_area_points_all[0][:][0][0, dic]) * scaling
                        try:
                            cor_values += x_data[0]
                        except (NameError, ValueError):
                            x_data, _, _ = load_forces(current_name=current_image_folder, window_size_average=1)
                            if x_data is not None:
                                x_data = x_data * ((np.linalg.norm(
                                    np.array(correlation_area_points_all[-1][0][0]) -
                                    np.array(correlation_area_points_all[0][0][0])) * scaling)
                                                   / np.linalg.norm(x_data[-1] - x_data[0]))
                                cor_values += x_data[0]
                    else:
                        print("\n\033[33;1;21mWARRNING\033[0m\n\t - Špatně definovaný směr.")
                        return

                if dic == 2:
                    if axes == "tot" or axes == '2':
                        subregion_values = np.linalg.norm(centers[shift_index] - centers[0], axis=1) * scaling
                    elif axes == "tot2" or axes == '4':
                        subregion_values = (centers[shift_index] - centers[0]).reshape(-1, 2)
                        subregion_values = (subregion_values[:, 0] + subregion_values[:, 1]) * scaling
                    else:
                        print("\n\033[33;1;21mWARRNING\033[0m\n\t - Špatně definovaný směr.")
                        return
                else:
                    subregion_values = (centers[shift_index][:, dic] - centers[0][:, dic]) * scaling

                if dic == 1:
                    try:
                        subregion_values += x_data[0]
                    except (NameError, ValueError):
                        x_data, _, _ = load_forces(current_name=current_image_folder, window_size_average=1)
                        if x_data is not None:
                            x_data = x_data * ((np.linalg.norm(np.array(correlation_area_points_all[-1][0][0]) -
                                                               np.array(correlation_area_points_all[0][0][0]))
                                                * scaling) / np.linalg.norm(x_data[-1] - x_data[0]))
                            subregion_values += x_data[0]

                try:
                    min_value = min(np.min(subregion_values), np.min(cor_values))
                except TypeError:
                    min_value = np.min(subregion_values)
                try:
                    max_value = max(np.max(subregion_values), np.max(cor_values))
                except TypeError:
                    max_value = np.max(subregion_values)

            elif show_correlation_areas:
                if axes == "tot" or axes == '2':
                    cor_values = (np.linalg.norm(np.array(correlation_area_points_all[shift_index][:][0][0]) -
                                                 np.array(correlation_area_points_all[0][:][0][0]), axis=1) * scaling)
                elif axes == "tot2" or axes == '4':
                    cor_values = (np.array(correlation_area_points_all[shift_index][:][0][0]) -
                                  np.array(correlation_area_points_all[0][:][0][0])).reshape(-1, 2)
                    cor_values = (cor_values[:, 0] + cor_values[:, 1]) * scaling
                elif dic == 0:
                    cor_values = (np.array(correlation_area_points_all[shift_index][:][0][0, dic]) -
                                  np.array(correlation_area_points_all[0][:][0][0, dic])) * scaling
                elif dic == 1:
                    cor_values = (np.array(correlation_area_points_all[shift_index][:][0, dic]) -
                                  np.array(correlation_area_points_all[0][0][:][0, dic[0]])) * scaling
                    try:
                        cor_values += x_data[0]
                    except (NameError, ValueError):
                        x_data, _, _ = load_forces(current_name=current_image_folder, window_size_average=1)
                        if x_data is not None:
                            x_data = x_data * ((np.linalg.norm(
                                np.array(correlation_area_points_all[-1][0][0]) -
                                np.array(correlation_area_points_all[0][0][0])) * scaling)
                                               / np.linalg.norm(x_data[-1] - x_data[0]))
                            cor_values += x_data[0]
                else:
                    print("\n\033[33;1;21mWARRNING\033[0m\n\t - Špatně definovaný směr.")
                    return
            else:
                print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatná definice zadání.")
                return

            """if min_val is None:
                try:
                    min_value = min(np.min(subregion_values), np.min(cor_values))
                except TypeError:
                    min_value = np.min(subregion_values)
            if max_val is None:
                try:
                    max_value = max(np.max(subregion_values), np.max(cor_values))
                except TypeError:
                    max_value = np.max(subregion_values)"""

            subregions_coords = coordinates[background_index]

            if heat_graph_title is None and dic is not None:
                heat_title = (f"Image {background_index}:  -  shift in direction:  '{axis[dic]}'\n"
                              f"[image {shift_index}: {image_files[shift_index]}]")
            elif heat_graph_title is not None and figures == 2:
                heat_title = heat_graph_title[dic]
            else:
                heat_title = heat_graph_title

            ax.set_title(heat_title, pad=10, wrap=True)

            ax.imshow(img, aspect='equal')

            arrow_length = min(img.shape[0] * 0.2, img.shape[1] * 0.2)
            # Určete body polygonu pro šipku
            points = np.int16([
                (0, 0), (arrow_length, 0), (arrow_length * 0.6, arrow_length * 0.3),
                (arrow_length * 0.6, arrow_length * 0.15), (arrow_length * 0.15, arrow_length * 0.15)])
            points = np.vstack((points, (points[:, [1, 0]])[:0:-1]))

            # Vykreslete šipku jako polygon
            ax.add_patch(Polygon(points, closed=True, facecolor='black', edgecolor='white', lw=1.5, alpha=0.6))
            for i in ((0.85, 0.45, 'x'), (0.25, 1.1, 'y')):
                ax.text(arrow_length * i[0], arrow_length * i[1], i[2], fontweight='bold', color='white',
                        fontsize=max(arrow_length // 65, 8),
                        bbox=dict(facecolor='black', edgecolor='white', boxstyle='round', pad=0.2, alpha=0.6))

            ax.set_facecolor(main_ax_rgba)
            # ax.axis('equal')
            ax.axis('off')

            # Nastavení omezení osy x a y
            ax.set_xlim(0, img.shape[1])  # Omezení osy x
            ax.set_ylim(img.shape[0], 0)  # Omezení osy y

            # Přidání stupnice barev (colorbar)
            scalar_map = plt.cm.ScalarMappable(cmap=str(colorbar_style))

            # scalar_map.set_array(subregion_values)
            scalar_map.set_clim(vmin=min_value, vmax=max_value)

            if min_value == max_value:
                colorbar_spacing = 1

            cax = make_axes_locatable(ax).append_axes("right", 0.2, pad=0.25)

            cbar = plt.colorbar(scalar_map, cax=cax, ticks=np.linspace(min_value, max_value, colorbar_spacing).tolist())
            # , pad=0.03, shrink=0.95, aspect=15)

            # Změna formátu
            tick_labels = cbar.get_ticks().tolist()
            tick_labels = [f"{label:.2f}".replace('.', ',') for label in sorted(tick_labels)]
            cbar.set_ticklabels(tick_labels)

            # Nastavení pozice značek na vnitřní a vnější stranu barevného pruhu
            cbar.ax.yaxis.set_ticks_position('right')  # 'both' - obě strany
            cbar.ax.tick_params(which='both', direction='out', color="gray", width=1, size=5, pad=5)  # 'inout'
            cbar.outline.set_edgecolor('black')
            cbar.outline.set_linewidth(1)

            if colorbar_label is not None:
                cbar.set_label(str(colorbar_label), labelpad=-15, y=1.08, rotation=0)

            # cbar.ax.plot([0, 1], [0, 0], color='black', lw=0.8, alpha=0.5, linestyle='dashed', dashes=[5, 8])
            for i in ([0, 0.25], [0.75, 1]):
                cbar.ax.plot(i, [0, 0], color='black', lw=0.6, alpha=0.5)

            """# Vykreslení podoblastí a jejich hodnot jako heatmapy
            for i, subregion_coords in enumerate(subregions_coords):
                # subregion_value = subregion_values[i]
                # Normalizace hodnoty pro lepší zobrazení barev na heatmapě
                # normalized_value = (subregion_value - min(subregion_values)) /
                #                      (max(subregion_values) - min(subregion_values))

                # Vybrání vhodné barvy pro heatmapu
                # heatmap_color = plt.cm.jet(subregion_values[i])
                heatmap_color = scalar_map.to_rgba(subregion_values[i])

                subregion_patch = Polygon(subregion_coords, closed=True, alpha=0.75, facecolor=heatmap_color,
                                          edgecolor='none')
                ax.add_patch(subregion_patch)"""
            # Vykreslení podoblastí a jejich hodnot jako heatmapyg
            [ax.add_patch(Polygon(subregion_coords, facecolor=scalar_map.to_rgba(subregion_values[i]), edgecolor='none',
                                  closed=True, alpha=0.75)) for i, subregion_coords in enumerate(subregions_coords)]

            if cor_values is not None:
                [ax.add_patch(
                    Rectangle((subregion_coords[0]), abs(subregion_coords[1, 0] - subregion_coords[0, 0]),
                              abs(subregion_coords[1, 1] - subregion_coords[0, 1]), edgecolor='none',
                              facecolor=scalar_map.to_rgba(cor_values), alpha=0.75)) for subregion_coords in
                    correlation_area_points_all[background_index]]

                """subregion_coords = correlation_area_points_all[background_index][0]
                ax.add_patch(Rectangle((subregion_coords[0]), abs(subregion_coords[1, 0] - subregion_coords[0, 0]),
                                       abs(subregion_coords[1, 1] - subregion_coords[0, 1]), edgecolor='none',
                                       facecolor=scalar_map.to_rgba(cor_values), alpha=0.75))"""

    # plt.subplots_adjust(left=0.01, right=0.965, top=0.94, bottom=0.035)
    plt.subplots_adjust(top=0.8,  # 0.83
                        bottom=0.1,  # 0.01
                        left=0.01,
                        right=1.0,
                        hspace=0.0,
                        wspace=0.0)

    fig.set_facecolor(figure_rgba)

    if get_graph:
        return fig, axs
    elif save_graph or save_graph_separately:
        graph_format = str(graph_format).replace(".", "")
        if graph_format not in ('jpg', 'jpeg', 'JPG', 'eps', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz',
                                'tif', 'tiff', 'webp'):
            print(f"\nZadaný nepodporavaný formát [ *.{graph_format} ], automaticky změněno na [ *.pdf ]")
            graph_format = 'pdf'
        if saved_graph_name is None:
            saved_graph_name = f"Displacement_Image_{shift_index:03d}_{image_files[shift_index].split('.')[0]}"
            saved_graph_name = saved_graph_name.replace(".", "_") + "." + graph_format
        elif saved_graph_name.lower().endswith("."):
            saved_graph_name = saved_graph_name[:-1].replace(".", "_") + "." + graph_format
        else:
            saved_graph_name = saved_graph_name.replace(".", "_") + "." + graph_format

        if save_graph_separately and figures > 1:
            import matplotlib.transforms as mtransforms

            saved_graph_name, graph_format = saved_graph_name.split(".")

            l1, b1, w1, _ = axs[0].get_position().bounds
            l2, _, _, _ = axs[1].get_position().bounds

            cor2 = [[l2 + (l2 - w1) * 2, b1 * 0.7], [l2 + w1, 1 - b1]]
            if make_line_graph:
                cor1 = [[0, b1 * 0.7], [l2 + (l2 - w1) * 2, 1 - b1]]
            else:
                cor1 = [[l1 + (l2 - w1) * 2, b1 * 0.7], [l1 + w1, 1 - b1]]

            # Uložení jednotlivých os jako obrázků
            for i, cor in enumerate((cor1, cor2)):
                saved_graph_path = os.path.join(current_folder_path, f'{saved_graph_name}_{i + 1}.{graph_format}')
                fig.suptitle("")

                fig.savefig(
                    "_.raw",
                    bbox_inches=mtransforms.Bbox([[0, 0], [0.0009, 0.003]]).transformed(
                        fig.transFigure - fig.dpi_scale_trans))

                fig.savefig(
                    saved_graph_path,
                    bbox_inches=mtransforms.Bbox(cor).transformed((fig.transFigure - fig.dpi_scale_trans)))

            os.remove("_.raw")
        else:
            plt.savefig(os.path.join(current_folder_path, saved_graph_name), format=graph_format, dpi=save_dpi,
                        bbox_inches='tight')

        plt.close("Final displacement graph")
    else:
        plt.pause(0.5)
        plt.show(block=block_graph)
        plt.pause(2)


def create_video_from_images(image_folder, output_video_path, fps=30, frame_width=1920, frame_height=1080,
                             video_length=None, codec='none'):
    if (frame_width, frame_height) not in ((640, 480), (1280, 720), (1920, 1080), (1440, 1080), (2560, 1440),
                                           (2048, 1080), (3840, 2160), (7680, 4320)):
        print("\nŠpatné rozlišení videa.")
        return

    if (frame_width, frame_height) in ((3840, 2160), (7680, 4320)):
        codec = 'H264'

    images = [img for img in os.listdir(image_folder) if
              img.endswith(('.jpg', '.jpeg', '.JPG', '.png', '.tif', '.tiff', '.webp'))]
    if not images:
        print(f"\n\tVideo nebylo vytvořeno, složka {image_folder} je prázdná.\n")
        return
    images.sort()

    if video_length is not None:
        fps = len(images) / video_length

    # Získáme rozměry prvního obrázku (předpokládáme, že všechny mají stejné rozměry)
    image_height, image_width = cv2.imread(os.path.join(image_folder, images[0]), 0).shape[:2]

    if output_video_path.lower().endswith(".mp4"):
        if codec == 'none':
            codec = 'H265'
        elif codec not in ('H264', 'X264', 'H265', 'VP90', 'mp4v', 'DIVX', 'XVID', 'FMP4', 'avc1'):
            print("Nepodporovaný codec pro mp4.")
            return
    elif output_video_path.lower().endswith(".avi"):
        if codec == 'none':
            codec = 'DIVX'
        elif codec not in ('DIVX', 'XVID', 'MJPG', 'WMV1', 'WMV2', 'mpg1', 'I420', 'IYUV', 'H264'):
            print("Nepodporovaný codec pro mp4.")
            return
    else:
        print("Nepodporovaný nebo nezadaný formát.")
        return

    # Nastavení kodeku a vytvoření objektu pro video zápis
    #                                   VideoWriter objekt s nekomprimovaným kodekem (VYUY)
    # fourcc = cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V') # - nefunkční
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    # Výpočet poměru šířky a výšky načteného obrázku
    image_ratio = image_width / image_height

    # Výpočet poměru šířky a výšky požadovaného pole
    desired_ratio = frame_width / frame_height

    # Rozhodnutí o změně velikosti obrázku
    if image_ratio > desired_ratio:
        new_width = frame_width
        new_height = np.int16(frame_width / image_ratio)
    else:
        new_width = np.int16(frame_height * image_ratio)
        new_height = frame_height

    # Umístění obrázku do pole, aby zabíralo maximální plochu
    x_offset = (frame_width - new_width) // 2
    y_offset = (frame_height - new_height) // 2

    for image_name in images:
        image = cv2.imread(os.path.join(image_folder, image_name), 1)

        # Změna velikosti obrázku
        resized_image = cv2.resize(image, (new_width, new_height))

        # Vytvoření prázdného pole
        output_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Umístěte fotografii na střed obrazu
        output_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        out.write(output_image)
    # Uzavření objektu pro video zápis
    out.release()
    print(f"\n\tVideo bylo vytvořeno do složky {output_video_path}\n")


def load_forces(current_name: str | bytes, zero_stage: int = 10, window_size_average: int = 3,
                window_size_start: int = 5):
    csv_file_name = current_name + ".csv"

    print("\nNačítání naměřených dat zatížení.")

    def load_csv(path):
        import pandas as pd

        if not isinstance(path, zipfile.ZipExtFile):
            if not os.path.exists(path):
                if not os.path.exists(os.path.join(folder_measurements, "data_txt", current_name + ".txt")):
                    print(f'\n\033[33;1;21mWARRNING\033[0m\n\tSelhalo načtení dat.\n\tPOPIS: soubor TXT neexistuje.')
                    return None, None, None
                from export_data import create_csv
                create_csv(file_name=current_name, folder_path_load=os.path.join(folder_measurements, "data_txt"),
                           folder_path_save=os.path.join(folder_measurements, "data_csv"), os=os, pd=pd)

        df = pd.read_csv(path)  # DATAFRAME

        # Odečtení průměru od všech následujících hodnot v 2. a 3. sloupci
        df.iloc[zero_stage:, 1] -= df.iloc[:zero_stage, 1].mean()
        df.iloc[zero_stage:, 2] -= df.iloc[:zero_stage, 2].mean()

        # Načtení dat
        x_data = df.iloc[:, 0].values  # První sloupec jako osa x - posun
        y_data = - (df.iloc[:, 1].values + df.iloc[:, 2].values)  # - celková síla
        photo_indexes = df[df['Photos'].notna()].index

        # Klouzavý průměr s oknem šířky 5
        window_start = max(3, window_size_start)
        while True:
            try:
                cumulative_sum = np.cumsum(y_data)
                cumulative_sum[window_start:] = cumulative_sum[window_start:] - cumulative_sum[:-window_start]
                # Najděte kladná čísla
                positive_numbers = y_data[max(window_start - 2, 0):window_start + 1][
                    y_data[max(window_start - 2, 0):window_start + 1] > 0]
                min_positive = np.min(positive_numbers)
                # Porovnáme průměry 5 po sobě jdoucích čísel s hodnotami na daných pozicích
                condition = (cumulative_sum / window_start) < min_positive
                break
            except ValueError:
                if window_start > window_size_start + 50:
                    condition = [True]
                    print("\nDosažení limitu pro hledání počátku měření")
                    break
                window_start += 1

        # Najdeme pozice, kde podmínka platí
        start_position = np.where(condition)[0][-1]
        x_data = x_data - x_data[start_position]  # Stanovení 0 pozice zatěžovnání

        # Průměrování dat
        # window_size = Velikost klouzavého okna
        weights = np.ones(window_size_average) / window_size_average
        extended_data = np.concatenate((y_data, np.repeat(y_data[-1], window_size_average)))
        smoothed_data = np.convolve(extended_data, weights, mode='valid')
        smoothed_data = smoothed_data[:-1]

        # del sys.modules['pandas']  ##################################################  ????????
        return x_data, smoothed_data, photo_indexes

    if saved_file_exist and load_calculated_data:
        try:
            # Funkce pro načtení fotografie z zip souboru
            path_to_zip_file = os.path.join(current_folder_path, saved_data_name + ".zip")

            with zipfile.ZipFile(path_to_zip_file, "r") as zip_obj:
                # Načítání souboru CSV z ZIP
                with zip_obj.open(csv_file_name) as file:
                    distances, forces, photo = load_csv(file)
        except KeyError as ke:
            try:
                path_to_csv = os.path.join(folder_measurements, "data_csv", csv_file_name)
                print(f'\n\033[33;1;21mWARRNING\033[0m\n\tV uložených datech se nenachází soubor: "{csv_file_name}"'
                      f'\n\tPOPIS: {ke}\n\t\t➤ Pokus o načtení souboru ze složky: '
                      f'[{os.path.join(folder_measurements, "data_csv")}]')
                # Načtení CVS z dané cesty
                distances, forces, photo = load_csv(path_to_csv)
                print(f' - Úspěšné načtení dat')
            except Exception as e:
                print(f'\n\033[33;1;21mWARRNING\033[0m\n\tSelhalo načtení dat.\n\tPOPIS: {e}')
                """program_shutdown(f'\n\033[31;1;21mERROR\033[0m'
                                 f'\n\tSelhalo načtení dat.\n\tPOPIS: {e}\n\t\t\033[41;30m➤ Ukončení programu.\033[0m',
                                 try_save=False)"""
                return None, None, None
    else:
        # Načtení CVS z dané cesty
        distances, forces, photo = load_csv(os.path.join(folder_measurements, "data_csv", csv_file_name))

    print("\tUkončení načítání dat.")

    return distances, forces, photo


# Funkce pro načtení jedné zvolené fotografie ze zip souboru
def load_photo(img_index: int, color_type: int = 1, give_path=False, photo_path=None):
    img = None

    def try_load_photo():
        # Načtení fotografie z dané cesty
        original_log_level = cv2.getLogLevel()  # Uložení původní úrovně logování
        cv2.setLogLevel(0)  # Nastavení úrovně logování na varování
        image = None
        try:
            img_path = os.path.join(current_path_to_photos, image_files[img_index])
            if give_path:
                return img_path
            image = cv2.imread(img_path, color_type)
            if image is None:
                raise MyException(f'\n\033[31;1;21mERROR\033[0m\n\tSelhalo načtení uložené fotografie [{img_path}]'
                                  f'\n\t\tJe doporučeno soubory zkontrolovat'
                                  f'\n\n\t\t\033[41;30m➤ Ukončení programu.\033[0m')
        except MyException as ex:
            program_shutdown(ex)
        finally:
            # Vrácení původní úrovně logování
            cv2.setLogLevel(original_log_level)
        return image

    if preload_photos:
        try:
            global preloaded_images
            if img_index > len(preloaded_images) - 1 or img_index < -len(preloaded_images):
                program_shutdown("Špatný index pro načtení fotografie")
            return preloaded_images[img_index].copy()
        except (NameError, Exception):
            if os.path.exists(photo_path):
                return cv2.imread(photo_path, color_type)
            else:
                return try_load_photo()
    elif photo_path is not None and os.path.exists(photo_path):
        img = cv2.imread(photo_path, color_type)
    elif saved_file_exist and load_calculated_data:
        try:
            # Funkce pro načtení fotografie z zip souboru
            path_to_zip_file = os.path.join(current_folder_path, saved_data_name + ".zip")
            zip_folder_name = "image_folder"

            # Funkce pro extrakci čísla z názvu souboru
            def number_extraction(string, path):
                num = (string.split(f"{path}/")[1]).split("_")[1].split(".")[0]  # 1. split = name, 2. split = number
                return np.int32(num)

            zip_files = []
            with zipfile.ZipFile(path_to_zip_file, "r") as zip_obj:

                for file_info in zip_obj.namelist():
                    if file_info.startswith(zip_folder_name + '/'):
                        zip_files.append(file_info)
                zip_files = [f for f in zip_files[1:] if f.endswith(photos_types)]

                sorted_indexes = np.argsort([number_extraction(s, zip_folder_name) for s in zip_files])
                # [::-1] => sestupně
                zip_files = np.array(zip_files)[sorted_indexes]
                if len(zip_files) == 0:
                    """print('\t\033[37;1;21mWARRNING\033[0m\033[37m:  V souboru ZIP nejsou uloženy fotografie.'
                          f'\t➤ Pokus o načtení fotografie ze složky [{current_image_folder}]\033[0m')"""
                    raise MyException(Exception)
                elif img_index >= len(zip_files):
                    path_to_zipped_image = None
                    raise ValueError("Fotografie s daným indexem nebyla nalezena.")
                else:
                    path_to_zipped_image = zip_files[img_index]

                if give_path:
                    raise MyException(Exception)
                    # return path_to_zipped_image

                with zip_obj.open(path_to_zipped_image) as photo:
                    img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), color_type)
        except (ValueError, KeyError) as e:
            print(f'\n\033[33;1;21mWARRNING\033[0m\n\tSelhalo načtení uložené fotografie [{path_to_zipped_image}]'
                  f'\n\tPOPIS: {e}\n\t\t➤ Pokus o načtení fotografie ze složky: [{current_path_to_photos}]')
            img = try_load_photo()
        except MyException:
            img = try_load_photo()
    else:
        img = try_load_photo()

    return img


def get_files_from_zipped_folder(zip_folder_name="image_folder"):
    """if 'zipfile' not in sys.modules:
        import zipfile"""

    path_to_zip_file = os.path.join(current_folder_path, saved_data_name + ".zip")

    zip_files = []
    files_names = []
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_obj:
        for file_info in zip_obj.namelist():
            if file_info.startswith(zip_folder_name + '/'):
                zip_files.append(file_info)
    for name in zip_files[1:]:
        files_names.append(name.split(f"{zip_folder_name}/")[1])
    zip_obj.close()
    if not files_names:
        print("\n\033[1;21mWARRNING\033[0m:  Chyba v načtení fotek ze souboru ZIP.")
        files_names = get_photos_from_folder(current_path_to_photos)
    return files_names


def get_photos_time_stamps(method="Auto", load_measurement_input=True):
    """method = 'Auto' | 'Original photos' | 'Zipped photos'"""
    global photos_times

    try_get_value, try_get_original_photos, try_get_zipped_photos = False, False, False
    if method == "Auto":
        try_get_value = try_get_original_photos = try_get_zipped_photos = True
    elif method == "Original photos":
        try_get_original_photos = True
    elif method == "Zipped photos":
        try_get_zipped_photos = True
    else:
        print("\nŠpatně zvolen typ načtení časů fotografií, bude použito automatické rozhodnutuní.")
        try_get_value = try_get_original_photos = try_get_zipped_photos = True

    times = []
    input_sequence = None

    if 'photos_times' in globals() and isinstance(photos_times, list) and len(photos_times) > 0 and try_get_value:
        try:
            times = photos_times.copy()
        except (ValueError, Exception):
            times = []
    if not times and try_get_original_photos:
        try:
            times = [os.path.getmtime(os.path.join(current_path_to_photos, i)) for i in image_files]
            times = [0] + [int(times[i + 1] - times[i]) for i in range(len(times) - 1)]
        except (ValueError, Exception):
            times = []
    if not times and try_get_zipped_photos:
        try:
            times = [np.int64(time.mktime(
                zipfile.ZipFile(os.path.join(current_folder_path, saved_data_name + ".zip"), 'r').getinfo(
                    file).date_time + (0, 0, 0)) - 3600) + 1 for file in
                     [name for name in zipfile.ZipFile(os.path.join(current_folder_path, saved_data_name + ".zip"),
                                                       'r').namelist() if name.startswith("image_folder/")][1:]]
            times = [0] + [int(times[i + 1] - times[i]) for i in range(len(times) - 1)]
        except (ValueError, Exception, MyException):
            times = []

    if load_measurement_input:
        # Otevřít textový soubor pro čtení časového nastavení
        if os.path.isfile(os.path.join(folder_measurements, "data_txt", current_image_folder + ".txt")):
            try:
                with open(os.path.join(folder_measurements, "data_txt", current_image_folder + ".txt"), 'r') as txt:
                    input_sequence = txt.readline().split()
                    txt.close()

                if input_sequence[0] == 'MMDIC' or input_sequence[0] == 'MMDIC2':
                    input_sequence = np.int16(input_sequence[-1])
                else:
                    print("\n\t\033[33;1;21mWARRNING\033[0m"
                          "\n\t\t - V souboru se nenachází zadávací sekvence 'MMDIC' nebo 'MMDIC2'.")
            except (ValueError, Exception):
                input_sequence = None

        return times, input_sequence
    else:
        return times


def save_data(data1_variables: list = None, data2_correlation: list = None, data2_rough_detect: list = None,
              data2_fine_detect: list = None, data2_point_detect: list = None, data3_additional_variables: dict = None,
              data_json: list | dict = None, current_measurement: str = "", file_name: str = "calculated_data",
              temporary_save: bool = False, save_photos: bool = False, compression_type: str = None):
    """Save data to zip file"""

    """if 'h5py' not in sys.modules:
        import h5py
    if 'zipfile' not in sys.modules:
        import zipfile
    if 'json' not in sys.modules:
        import json"""

    if temporary_save:
        compression_type = 'gzip'
    elif compression_type not in ('gzip', 'lzf', None):
        print("\n\033[1;21mWARRNING\033[0m:  Chyba v zadání typu komprese.")
        compression_type = None

    # Check
    if all(d is None for d in [data1_variables, data2_correlation, data2_rough_detect, data2_fine_detect,
                               data2_point_detect, data_json]):
        raise SaveError("\n\033[33;1;21mWARRNING\033[0m\n\tK uložení nejsou zvolena žádná data.")

    print("\nUkládání spočítaných dat.")

    if not temporary_save:
        report_path = os.path.join(current_folder_path, make_final_report())
        log_file_path = os.path.join(current_folder_path, 'outputs.log')
        csv_data_path = os.path.join(folder_measurements, "data_csv", current_measurement + ".csv")
    else:
        report_path = log_file_path = csv_data_path = ""
    data_file_path = os.path.join(current_folder_path, 'data.h5')
    settings_file_path = os.path.join(current_folder_path, 'settings.json')

    image_zip_folder_name = "image_folder"

    # Uložení obrázků do zip archivu
    file_name = file_name.replace(".zip", "").replace(".", "_")
    zip_file_name = os.path.join(current_folder_path, file_name + ".zip")
    if os.path.exists(zip_file_name):
        print(f"\nSoubor [{file_name + '.zip'}] již existuje.\n\tChcete ho opravdu nenávratně přepsat?")
        while True:
            if temporary_save:
                ans = "Y"
            elif super_speed:
                ans = "N"
            else:
                ans = askstring("Přepsání souboru.",
                                f"\nSoubor [{file_name + '.zip'}] již existuje."
                                "\n\tChcete ho opravdu nenávratně přepsat?\nZadejte Y / N: ")
                # ans = input("\t\tZadejte Y / N: ")
            if ans == "Y":
                print("\n\tZvolena možnost 'Y'\n\tSoubor bude přepsán.")
                break
            elif ans == "N":
                if super_speed:
                    new_name = file_name + "_new.zip"
                else:
                    new_name = askstring("Jméno souboru", "Zadejte nové jméno: ")
                    # new_name = str(input("\n\tZvolena možnost 'N'\n\t\tZadejte nové jméno:"))
                print(f"\n\t\tZvolena možnost 'N'\nNové jméno: ' {new_name} '")
                if os.path.exists(os.path.join(current_folder_path, new_name)):
                    new_name = file_name + f"_new_{get_current_date(date_format='%H-%M-%S_%d-%m-%Y')}.zip"
                zip_file_name = os.path.join(current_folder_path, new_name)
                break
            else:
                print("\n Zadejte platnou odpověď.")

    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # zipfile.ZIP_STORED / zipfile.ZIP_DEFLATED / zipfile.ZIP_BZIP2 / zipfile.ZIP_LZMA

        # Uložení .h5 souboru pomocí Blosc komprese do zip archivu
        with h5py.File(data_file_path, 'w') as file:

            if not isinstance(data1_variables, list):
                data1_variables = [data1_variables]

            dicts_list = [i for i, j in enumerate(data1_variables) if isinstance(j, dict)]
            for i, j in enumerate(dicts_list):
                status_group = file.create_group(f'dictionary_{i}')
                for key, value in data1_variables[j].items():
                    status_group.attrs[key] = value

            data1_variables = [j for i, j in enumerate(data1_variables) if i not in dicts_list]

            # Vytvoření skupiny pro uložení proměnných
            data_group = file.create_group('variables')  # Skupina daty
            [data_group.create_dataset(f'var{i:05d}', data=variable,
                                       compression=compression_type if isinstance(variable, np.ndarray) and len(
                                           variable.shape) >= 2 else None)
             for i, variable in enumerate(data1_variables)]

            var_group = file.create_group('additional_variables')
            if isinstance(data3_additional_variables, dict):
                [var_group.create_dataset(f'{key}', data=value,
                                          compression=compression_type if isinstance(value, np.ndarray) and len(
                                              value.shape) >= 2 else None) for
                 key, value in data3_additional_variables.items()]

            data2 = dict(data_correlation=data2_correlation, data_rough_detect=data2_rough_detect,
                         data_fine_detect=data2_fine_detect, data_point_detect=data2_point_detect)

            for data_name, data_type in data2.items():
                if data_type is not None:
                    if not isinstance(data_type, list):
                        data_type = [data_type]

                    big_all_data_group = file.create_group(data_name)
                    [
                        [
                            big_all_data_group.create_dataset(f'subgroup_{i:05d}/data_{j:05d}', data=array,
                                                              compression=compression_type
                                                              if isinstance(array, np.ndarray) and len(
                                                                  array.shape) >= 2 else None)
                            for j, array in enumerate(sublist)
                        ]
                        for i, sublist in enumerate(data_type)
                    ]

            """# Skupina s fotkami
            photos_group = file.create_group('photos')
            photos_set = [photos_group.create_dataset(f'photo{i}',
                                                      data=cv2.imread(os.path.join(current_path_to_photos, photo), 1),
                                                      compression='gzip') for i, photo in enumerate(data12)]
            # compression='lzf' / compression='gzip'

            photos_names = [photos_group.create_dataset(f'photo{i}_name', data=str(name)) for
                            i, name in enumerate(data2_correlation)]"""

        file.close()

        # Uložení fotek do ZIPu
        if not temporary_save or save_photos:
            zipf.writestr(f"{image_zip_folder_name}/", '')
            # Přesun fotografií do složky uvnitř zip souboru
            for photo in image_files:
                path_to_photo = os.path.join(current_path_to_photos, photo)
                zipf.write(path_to_photo, f"{image_zip_folder_name}/{photo}", zipfile.ZIP_STORED)

        # Ukládání nastavení do JSON souboru
        if data_json is not None:
            with open(settings_file_path, 'w') as file:
                json.dump(data_json, file)
            file.close()
            zipf.write(settings_file_path, 'settings.json', zipfile.ZIP_DEFLATED)  # file_path , arcname, compression
            os.remove(settings_file_path)

        if not temporary_save and os.path.exists(log_file_path):
            try:
                zipf.write(log_file_path, 'outputs.log', zipfile.ZIP_DEFLATED)
            except FileNotFoundError:
                pass

        zipf.write(data_file_path, 'data.h5', zipfile.ZIP_DEFLATED)
        if os.path.exists(os.path.join(current_folder_path, 'areas.json')):
            zipf.write(os.path.join(current_folder_path, 'areas.json'), 'areas.json', zipfile.ZIP_DEFLATED)

        if current_measurement is not None and not temporary_save and os.path.exists(csv_data_path):
            zipf.write(csv_data_path, current_measurement + ".csv", zipfile.ZIP_DEFLATED)
        if not temporary_save and os.path.exists(report_path):
            zipf.write(report_path, 'final_output_report.txt', zipfile.ZIP_DEFLATED)

    zipf.close()

    # Odstranění původních souborů
    os.remove(data_file_path)
    if not temporary_save and os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
        except FileNotFoundError:
            pass


def load_data():
    """Load data from zip file"""

    """if 'h5py' not in sys.modules:
        import h5py
    if 'zipfile' not in sys.modules:
        import zipfile
    if 'json' not in sys.modules:
        import json"""

    print("\nNačítání uložených dat.")

    zip_file_name = os.path.join(current_folder_path, saved_data_name + ".zip")
    try:
        # Načtení dat z zip archivu
        with zipfile.ZipFile(zip_file_name, 'r') as zipf:

            file_list = zipf.namelist()

            # Zjištění, zda je zip soubor prázdný
            if not file_list:
                raise MyException(f"\033[31;1;21mError:\033[0m Zip file [{zip_file_name}] is empty.")

            try:
                # Načítání nastavení z JSON souboru
                with zipf.open('settings.json') as file:
                    loaded_settings = json.load(file)

                    if not (loaded_settings['calculations_statuses']['Correlation'] and
                            (loaded_settings['calculations_statuses']['Rough detection'] is False and
                             loaded_settings['calculations_statuses']['Point detection'] is False)):
                        # loaded_settings.popitem()
                        compare_settings = loaded_settings.copy()

                        if compare_settings['all_photos']:
                            compare_settings['end'] = "all"

                        del compare_settings['scale'], compare_settings['calculations_statuses']
                        # Porovnání načteného nastavení s aktuálními hodnotami
                        if len(compare_settings) != len(settings):
                            print("\t\t- Uložené nastavení je odlišné od aktuálního."
                                  "\n - Data nejsou kompatibilní.\t(Pravděpodobně jiná verze programu.)")
                            raise MyException("\n\033[31;1;21mWARRNING\033[0m "
                                              "Saved values in 'json' file [settings.json] has different lenght."
                                              f"\n- {'Nastavené:':<10} {settings}\n- "
                                              f"{'Načtené:':<10} {compare_settings}")

                        if compare_settings != settings:
                            print("\n\033[31;1;21mWARRNING\033[0m: "
                                  "Saved values in 'json' file [settings.json] are different."
                                  f"\n- {'Nastevené:':<10} {settings}\n- {'Načtené:':<10} {compare_settings}"
                                  "\t\t- Uložené nastavení je odlišné od aktuálního."
                                  "\n\tChcete provést nový výpočet?")

                            while True:
                                if super_speed:
                                    ans = "N"
                                else:
                                    ans = askstring("Chcete provést nový výpočet?",
                                                    "Chcete provést nový výpočet?\nZadejte Y / N: ")
                                    # ans = input("\t\tZadejte Y / N: ")
                                if ans == "Y":
                                    print("\n\tZvolena možnost 'Y'")
                                    raise MyException("\nZahájení nového výpočtu.")
                                elif ans == "N":
                                    print("\n\tZvolena možnost 'N'\n\t- Budou použity načetné hodnoty.")
                                    break
                                else:
                                    print("\n Zadejte platnou odpověď.")
                        else:
                            print("\t\t- Uložené nastavení je stejné s aktuálním.")
                    else:
                        print("\nNačtená data odpovídají pouze datům z korelace.")

                file.close()
            except (Exception, MyException) as ke:
                print(f'\n\033[33;1;21mWARRNING\033[0m'
                      f'\n\tChyba načtení nastavení JSON\n\tPOPIS: {ke}\n\n- Použití aktuálního nastavení.')

            # Načtení .h5 souboru
            with zipf.open('data.h5') as h5_file:
                with h5py.File(h5_file, 'r') as file:
                    # Seznam skupin v souboru
                    group_names = list(file.keys())

                    # Načtení skupiny, ve které jsou uložené proměnné
                    """photos_group = file['photos']"""

                    data = None
                    if 'variables' in group_names:
                        data_group = file['variables']
                        # Načtení jednotlivých proměnných z datasetů a uložení do seznamu
                        data = [data_group[f'var{i:05d}'][()] for i in range(len(data_group))]

                        # Slovník statusů (atributů)
                        for d in [key for key in file.keys() if key.startswith('dictionary_')]:
                            data += [{key: value for key, value in file[d].attrs.items()}]

                    dataset_variables = {}
                    if 'additional_variables' in group_names:
                        dataset_variables = {key: value[:] for key, value in file['additional_variables'].items()}

                    dataset_values = dict(data_correlation=None, data_rough_detect=None,
                                          data_fine_detect=None, data_point_detect=None)
                    for group_name in dataset_values.keys():
                        if group_name in group_names:
                            big_all_data_group = file[group_name]
                            dataset_values[group_name] = [
                                [
                                    dataset[:] for dataset in subgroup.values()
                                ]
                                for subgroup in big_all_data_group.values()
                            ]

                    """amount = np.int16(len(photos_group) / 2)
                    photos = [photos_group[f'photo{i}'][()] for i in range(amount)]
                    photo_names = [photos_group[f'photo{i}_name'][()].decode() for i in range(amount)]"""
                file.close()
            h5_file.close()

    except KeyError as ke:
        zipf.close()
        program_shutdown(f'\n\033[31;1;21mERROR\033[0m'
                         f'\n\tSelhalo načtení uložených dat\n\tPOPIS: {ke}'
                         f'\n\n\t\t\033[41;30m➤ Ukončení programu.\033[0m', try_save=False)
    zipf.close()
    return loaded_settings, data, dataset_values, dataset_variables  # , photos, photo_names


def dialog(error_message):
    global load_calculated_data, save_calculated_data, second_callout

    print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba v uložených datech.\n\t➤ POPIS: {error_message}")
    print("\nData nebyla správně načtena.\n\tChcete provést nový výpočet?\n")
    while True:
        if super_speed:
            ans = "Y"
        else:
            ans = askstring("Chcete provést nový výpočet?", "Chcete provést nový výpočet?\nZadejte Y / N: ")
            # ans = input("\t\tZadejte Y / N: ")
        if ans == "Y":
            print("\n\tZvolena možnost 'Y'")
            print("\n\t\tChcete uložit nově vypočtená data?\n")
            while True:
                if super_speed:
                    ans = "Y"
                else:
                    ans = askstring("Chcete uložit nově vypočtená data?",
                                    "Chcete uložit nově vypočtená data?\nZadejte Y / N: ")
                    # ans = input("\t\tZadejte Y / N: ")
                if ans == "Y":
                    print("\n\t\tZvolena možnost 'Y'\n\tBude proveden výpočet a data budou uložena."
                          "\n➤ Spouštění výpočtu.")
                    second_callout = True
                    break
                elif ans == "N":
                    print("\n\t\tZvolena možnost 'N'\n\tBude proveden výpočet bez uložení dat.\n➤ Spouštění výpočtu.")
                    break
                else:
                    print("\n Zadejte platnou odpověď.")
            load_calculated_data = False
            main()
            program_shutdown(try_save=False)
        elif ans == "N":
            program_shutdown("\n\tZvolena možnost 'N'\n\tZahájení ukončení programu.", try_save=False)
        else:
            print("\n Zadejte platnou odpověď.")


def make_angle_correction(image_to_get_angle=None, image_to_warp=None, points_to_warp=None):
    global width, height, angle_correction_matrix
    if image_to_get_angle is None:
        img = load_photo(0, 0)
    else:
        img = image_to_get_angle.copy()

    if 'width' not in globals() or 'height' not in globals():
        # Získání rozměrů obrázku
        height, width = img.shape[:2]

    if 'angle_correction_matrix' not in globals() or not isinstance(angle_correction_matrix, np.ndarray):
        # Vypočtěte střed obrázku
        center = (width // 2, height // 2)

        qr_path = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\templates\QR'

        mask = np.zeros((height, width), dtype=np.uint8)
        for _ in range(1, 5):
            qr_img_path = os.path.join(qr_path, f'QR_{_}.png')
            if os.path.isfile(qr_img_path):
                qr_img = cv2.imread(qr_img_path, 0)
                loc, h, w, _ = match(qr_img, img, tolerance=0.001, method=cv2.TM_CCOEFF_NORMED)
                if loc is not None:
                    mask[max(loc[1] - 100, 0):min(loc[1] + h + 100, height),
                    max(loc[0] - 100, 0):min(loc[0] + w + 100, width)] = 255

        masked_img = cv2.convertScaleAbs(cv2.bitwise_and(img, img, mask=mask), alpha=1.1, beta=-50)
        decoded_objects = qr_detect(masked_img)

        point1, point2, point3, point4 = None, None, None, None

        if decoded_objects and len(decoded_objects) >= 2:
            points = [None, None, None, None]
            for obj in decoded_objects:
                name = obj.data.decode('utf-8')
                if not name.startswith(".*CP*.") or ".*CP*." not in name:
                    continue
                points[int(name.replace(".*CP*._N#", "")) - 1] = np.mean(obj.polygon, axis=0)
            point3, point4, point1, point2 = points

        if (point1 is None and point2 is None) and not (point3 is None and point4 is None):
            point1, point2, point3, point4 = point3, point4, None, None

        if (point1 is None and point2 is None):
            print(f"\n\033[33;1;21mWARRNING\033[0m\n\tQR kódy nebyly nalezeny.")
            scale_paths = os.path.join(folder_measurements, "scale")
            template1 = cv2.imread(scale_paths + r"\start_1.png", 0)
            template2 = cv2.imread(scale_paths + r"\end_1.png", 0)
            point1 = match(template1, img)[0]
            point2 = match(template2, img)[0]

        angle_degrees1 = np.rad2deg(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
        # 0.2798794602553504 // 0.3498172464499053
        print(f"\n\033[93mÚhel pootočení stroje je: \033[95m{angle_degrees1} °\033[0m")

        if point3 is not None and point4 is not None:
            angle_degrees2 = np.rad2deg(np.arctan2(point4[1] - point3[1], point4[0] - point3[0]))
            print(f"\033[93mÚhel pootočení příčníku je: \033[95m{angle_degrees2} °\033[0m")
            print(f"\t\033[93mRozdíl úhlů je: \033[95m{angle_degrees1 - angle_degrees2} °\033[0m")

        angle_correction_matrix = cv2.getRotationMatrix2D(center, angle_degrees1, 1.0)

    """for point1, point2 in ((top_left1, top_left2), (points_pos[0], points_pos[-1]),
                           (points_track[0][0], points_track[5][0]), (points_track[2][0], points_track[3][0])):
        angle_rad = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        angle_degrees = np.degrees(angle_rad)
        print(f"\n\033[93mÚhel pootočení stroje je: \033[95m{angle_degrees} °\033[0m")"""

    try:
        if isinstance(image_to_warp, np.ndarray):
            # Aplikujte transformační matici na obrázek
            image_to_warp = cv2.warpAffine(image_to_warp, angle_correction_matrix, (width, height))
        if isinstance(points_to_warp, np.ndarray):
            # Aplikujte transformační matici na body
            points_to_warp = cv2.transform(points_to_warp.reshape(1, -1, 2), angle_correction_matrix).reshape(-1, 2)
    except (ValueError, Exception) as e:
        print(f"\nProblém s transformací bodů\n\tPOPIS: {e}")

    if isinstance(image_to_warp, np.ndarray) or isinstance(points_to_warp, np.ndarray):
        if isinstance(image_to_warp, np.ndarray) and isinstance(points_to_warp, np.ndarray):
            return image_to_warp, points_to_warp
        elif isinstance(image_to_warp, np.ndarray):
            return image_to_warp
        elif isinstance(points_to_warp, np.ndarray):
            return points_to_warp


def finalize_results_points(found_coordinates, x1, y1):
    def rotate_points(points, start__point, end__point):  # coordinates correction (oprava souřadnic)
        # TODO: pootočení souřadnic podle úhlu z corelation (svislost)
        angle_rad = np.arctan2(end__point[1] - start__point[1], end__point[0] - start__point[0]) + np.pi / 2

        # Rotuje body kolem počátku o daný úhel
        matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad), np.cos(angle_rad)]])

        return np.dot((points - start__point), matrix) + start__point  # transformed_points

    dist1, dist2, dist2_x, dist1_x, dist2_y, dist1_y = [0] * 7

    make_scale = False
    if make_scale and scale == 1:
        print("Chcete ručně zadat měřítko?")
        while True:
            make_scale = askstring("Chcete ručně zadat měřítko?", "Chcete ručně zadat měřítko?\nZadejte Y / N: ")
            # make_scale = input("\tZadejte Y nebo N: ")
            if make_scale == "Y":
                do_scale(auto_scale=False)
                while True:
                    print("\nChcete proces zopakovat?")
                    rerun = askstring("Chcete proces zopakovat?", "Chcete proces zopakovat?\nZadejte Y / N: ")
                    # rerun = input("\tZadejte Y nebo N: ")
                    if rerun == "Y":
                        do_scale(auto_scale=False)
                    elif rerun == "N":
                        break
                    else:
                        print("\nNeplatná odpověď. Zadejte pouze Y nebo N.")
                del rerun
                break

            elif make_scale == "N":
                do_scale(img=load_photo(img_index=0, color_type=0), auto_scale=True)
                break
            else:
                print("\nNeplatná odpověď. Zadejte pouze Y nebo N.")

    found_coordinates = found_coordinates * scale

    # rotate picture:
    rotation_matrix = photo = np.zeros((2, 2), dtype=np.float64)  # TODO ################
    start_point = np.float64((0, 0))
    corner_points = np.float64(((0, 0), (width, 0), (0, height), (width, height)))

    rotated_corner_points = np.dot((corner_points - start_point), rotation_matrix) + start_point

    rotation_matrix[0, 1], rotation_matrix[1, 0] = rotation_matrix[1, 0], rotation_matrix[0, 1]
    rotation_matrix_photo = np.column_stack((rotation_matrix, start_point))
    rotated_photo = cv2.warpAffine(photo, rotation_matrix_photo, (width, height))

    """else:
        dist1 = 100
        dist2 = 220
        dist1_x, dist1_y = 1521, 1457
        dist2_x, dist2_y = 4338, 1468

    x_f_mean = np.mean(found_coordinates[:, 0])
    y_f_mean = np.mean(found_coordinates[:, 1])

    dif_x = abs(x1 - x_f_mean)
    dif_y = abs(y1 - y_f_mean)

    dist_mm = dist2 - dist1
    dist_px = np.sqrt((dist2_x - dist1_x) ** 2 + (dist2_y - dist1_y) ** 2)

    final_dist_x_mm = dist_mm / dist_px * dif_x
    final_dist_y_mm = dist_mm / dist_px * dif_y

    print("\nScale:", round((dist_mm / dist_px), 3))

    print("\nVelikost fotografie:\t\t", height, "x", width, "px")
    print("Průměrný výškový rozdíl:\t\t", round(dif_y, 5), "px")
    print("Průměrný vodorovný rozdíl:\t\t", round(dif_x, 5), "px")

    print("\nPrůměrný výškový rozdíl:\t\t", round(final_dist_y_mm, 5), "mm")
    print("Průměrný vodorovný rozdíl:\t\t", round(final_dist_x_mm, 5), "mm")"""


def print_progress_bar(current_number: int | float, total_number: int | float, round_number: int = 5,
                       bar_length: int = 20, start_text=""):
    if current_number < total_number:
        progress = np.float32(current_number / total_number)  # min(current_number / total_number, 1.0)
        num_bars = np.int8(np.round(bar_length * progress))
        bar = "▰" * num_bars + "▱" * (bar_length - num_bars)
        percentage = f"{np.int8(np.round(progress * 100 / round_number) * round_number):3d}%"  # round progress
        text = f"{start_text}{bar}  {percentage}   [{current_number}/{total_number}]"
        print(f"\r{text}", end="")
    else:
        bar = "▰" * bar_length
        bar = f"{start_text}{bar}  100%   [𝘾𝙤𝙢𝙥𝙡𝙚𝙩𝙚𝙙: {total_number}]"  # "Basic: Math Sans Bolt Italic"
        print(f"\r{bar}", end="")
        print("")


def make_final_report():
    global correlation_area_points_all

    output_report_name = 'output_report_' + get_current_date() + '.txt'

    output_report_path = os.path.join(current_folder_path, output_report_name)

    if scale == 1:
        do_scale(load_photo(img_index=0, color_type=0))
    distance_correlation = np.linalg.norm(np.array(correlation_area_points_all[-1][0][0]) -
                                          np.array(correlation_area_points_all[0][0][0])) * scale
    data_distances, data_forces, photo_indexes = load_forces(current_name=current_image_folder, window_size_average=1)
    if any(data is None for data in (data_distances, data_forces, photo_indexes)):
        print("\n\033[33;1;21mWARRNING\033[0m\n\tFinální zpráva nevytvořena.")
        return ""
    data_distances = data_distances * (distance_correlation / np.linalg.norm(data_distances[-1] - data_distances[0]))
    max_index_force = np.argmax(data_forces)

    data = [
        {"comment": "Final distance [mm]", "value": round(distance_correlation, 3)},
        {"comment": "Final force [N]", "value": round(data_forces[-1], 3)},
        {"comment": "Distance at maximum force [mm]", "value": round(data_distances[max_index_force], 3)},
        {"comment": "Maximum force [N]", "value": round(data_forces[max_index_force], 3)},
        {"comment": "", "value": None},
    ]

    output_data = [f"Comment: {item['comment']}\n\tValue: {item['value']}\n\n" for item in data]

    with open(output_report_path, 'w') as txt_file:
        txt_file.writelines(output_data)
        txt_file.close()

    return output_report_name


def predict_time(folders):
    number_of_photos, duration, pc_status = 0, 0, 1

    # Definujte strukturu pro výsledek funkce GetSystemPowerStatus
    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_byte),
            ("BatteryFlag", ctypes.c_byte),
            ("BatteryLifePercent", ctypes.c_byte),
            ("Reserved1", ctypes.c_byte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    # Zavolejte funkci GetSystemPowerStatus a uložte výsledek do instance této struktury
    status = SystemPowerStatus()
    result = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))

    # Kontrola stavu napájení
    if result != 0:
        if status.ACLineStatus == 1:
            pass
        elif status.ACLineStatus == 0:
            pc_status = 2
        else:
            print("Stav napájení nelze zjistit.")
    else:
        print("Chyba při získávání informací o stavu napájení.")

    if do_just_correlation:
        duration += 5.5
    else:
        duration += 25  # - CORRELATION + ROUGH CALCULATION
        if do_calculations['Do Fine detection']:
            duration += 5  # - FINE CALCULATION
        if do_calculations['Do Point detection']:
            duration += 1.5  # - FIND POINTS

    for folder in folders:
        # Vytvoření cesty k cílovým fotografiím
        current_photo_folder = os.path.join(main_image_folder, folder, source_image_type[1])
        # Kontrola zda aktuální složka existuje a není prázdná a případně změna stránky
        if not any(f.endswith(photos_types) for f in os.listdir(current_photo_folder)):
            # Složka je prázná
            current_photo_folder = os.path.join(main_image_folder, folder, source_image_type[0])
            # Nová složka
            if not any(f.endswith(photos_types) for f in os.listdir(current_photo_folder)):
                continue
        try:
            number_of_photos += len(get_photos_from_folder(current_photo_folder))
        except TypeError:
            print(f"\n\033[33;1;21mWARRNING\033[0m:  Problém u načtení fotoggrafií složky: [ {folder} ]")
            continue

    if number_of_photos == 0:
        program_shutdown("\n\033[31;1;21mERROR\033[0m\n\tSložky jsou prázné.\n\tCelkový čas je 0 sekund."
                         "\n\t\t\033[41;30m➤ Ukončení programu.\033[0m", try_save=False)
    else:
        time_consumption = np.int32(number_of_photos * duration * pc_status)

        print(f"\nCelkový čas bude odhadem {time_consumption} sekund.\n"
              f"\t- celkem: {np.int32(time_consumption // 3600)} hodin, {np.int32((time_consumption % 3600) // 60)} "
              f"minut, {np.int32(time_consumption % 60)} sekund\nOdhad dokončení v:  "
              f"{time.strftime('%H:%M, %d.%m. %Y', time.localtime(time.time() + time_consumption))}\n")

    return


def try_save_data(zip_name="data_autosave", temporary_file=False, overwrite=False):
    global variable_names, all_photos

    if all(not value for value in do_calculations.values()):
        return

    try:
        dataset1 = [int(start), int(end), all_photos, main_image_folder, int(size), int(fine_size),
                    int(points_limit), float(precision), float(scale), calculations_statuses]

        if zip_name == saved_data_name and temporary_file:
            zip_name = saved_data_name + "_temp"

        if make_temporary_savings:
            print(f"\n\t\033[37m- Pokus o průběřné uložení dat do souboru: '{zip_name}.zip'\033[0m")
        else:
            print(f"\n- Pokus o uložení dat do souboru: '{zip_name}.zip'")

        if do_just_correlation:
            set_data = {name: value for name, value in zip((variable_names[0:3]), (dataset1[0:3]))}
        else:
            # Seskupení proměnných aktuálního nastavení do slovníku pro soubor JSON
            """settings = [item.tolist() if isinstance(item, np.ndarray) else
                       item for item in (data[0:2] + data[3:7])]"""
            set_data = {name: value for name, value in
                        zip(variable_names, (dataset1[0:3] + dataset1[4:]))}
            # {f'var{i}': variable for i, variable in enumerate(dataset)}

        if calculations_statuses['Correlation']:
            dataset2_correlation = correlation_area_points_all
        else:
            dataset2_correlation = None

        if calculations_statuses['Rough detection']:
            dataset2_rough_detect = [triangle_vertices_all, triangle_centers_all, triangle_indexes_all,
                                     wrong_points_indexes_all, end_marks_all, key_points_all]
        else:
            dataset2_rough_detect = None

        if calculations_statuses['Fine detection']:
            dataset2_fine_detect = [fine_triangle_points_all, fine_mesh_centers_all]
        else:
            dataset2_fine_detect = None

        if calculations_statuses['Point detection']:
            dataset2_point_detect = [tracked_points_all, tracked_rotations_all]
        else:
            dataset2_point_detect = None

        dataset_3 = dict(angle_correction_matrix=angle_correction_matrix, photos_times=photos_times)
        if all(val is None for val in dataset_3.values()):
            dataset_3 = None

        """if not temporary_file and overwrite:
            if (os.path.exists(os.path.join(current_folder_path, f"{zip_name}.zip"))
                    and make_temporary_savings):
                try:
                    os.remove(os.path.join(current_folder_path, f"{zip_name}.zip"))
                except PermissionError as e:
                    print(f"PermissionError: {e}")
                print("\t\t\033[37mDočasné soubory odstraněny\033[0m")"""

        # Uložení do souboru , uložení dat
        save_data(data_json=set_data,
                  data1_variables=[main_image_folder, current_path_to_photos, float(scale), photos_times],  # dataset1
                  data2_correlation=dataset2_correlation,
                  data2_rough_detect=dataset2_rough_detect,
                  data2_fine_detect=dataset2_fine_detect,
                  data2_point_detect=dataset2_point_detect,
                  data3_additional_variables=dataset_3,
                  current_measurement=current_image_folder, file_name=f"{zip_name}", temporary_save=temporary_file)
        if make_temporary_savings:
            print("\n\t\033[37m- Průběžná data úspěšně uložena.\033[0m")

    except (Exception, SaveError) as e:
        print(f"\n\033[31;1;21mERROR\033[0m"
              f"\n\tSelhání uložení dat:\n\t\tPOPIS: {e}\n- Data nebyla uložena.")

        return SaveError


def update_text_window(text: str, entry):
    entry.config(state="normal")  # Odblokování textového pole
    entry.delete("1.0", "end")  # Smazání existujícího textu
    entry.insert("1.0", f"\n{text}")  # Vložení nového textu
    entry.tag_configure("center", justify="center")
    # Přidání nových tagů pro tučný a větší text
    entry.tag_configure("large", font=("Helvetica", 14, "bold"))
    # Aplikace tagů na text
    entry.tag_add("large", "1.0", "end")
    entry.tag_add("center", "1.0", "end")
    entry.config(state="disabled")  # Blokování textového pole


def main():
    global image_files, gray1, gray2, width, height, saved_file_exist, auto_crop, scale, variable_names, all_photos, \
        current_path_to_photos, current_folder_path, current_image_folder, calculations_statuses, saved_data_name
    global start, end, main_image_folder, size, fine_size, points_limit, precision, settings, second_callout
    global points_pos, points_neg, points_cor, points_max, correlation_area_points_all, angle_correction_matrix
    global triangle_vertices_all, triangle_centers_all, triangle_indexes_all, triangle_points_all, \
        wrong_points_indexes_all, key_points_all, end_marks_all, photos_times, main_counter
    global fine_triangle_points_all, fine_mesh_centers_all, tracked_points_all, tracked_rotations_all

    def check_folder(folder, text1, text2, text3, important_folder=True):
        if not os.path.isdir(folder):
            program_shutdown(f"\n\033[31;1;21mERROR\033[0m\n\t{text1}: [{folder}] {text2}."
                             f"\n\t\033[41;30m➤ Ukončení programu.\033[0m", try_save=False)

        # Získání seznamu souborů ve složce
        folder_insides = [os.path.splitext(file)[0] for file in os.listdir(folder)]

        if isinstance(folder_insides[0], bytes):
            print(f"\n\033[33;1;21mWARRNING\033[0m:  Názvy jsou v bytech.")
            folder_insides = [item.decode('utf-8') for item in folder_insides if isinstance(item, bytes)]

        if len(folder_insides) == 0:
            if important_folder:
                program_shutdown(f"\n\033[31;1;21mERROR\033[0m\n\t{text1}: [{folder}] {text3}."
                                 "\n\t\033[41;30m➤ Ukončení programu.\033[0m", try_save=False)
            else:
                print(f"\n\033[34;1;21mWARRNING\033[0m\n\t{text1}: [{folder}] {text3}.")

        return folder_insides

    if dynamic_mode:
        images_folders = browse_directory(window_title="Vyberte složky s fotografiemi k výpočtu")
    else:
        images_folders = check_folder(main_image_folder, "Složka s fotkami", "neexistuje", "je prázdná")

    images_folders = [name for name in images_folders if name.startswith(data_type) or name.startswith(",")]
    # images_folders = images_folders[4:-2]  # TODO ############ potom změnit počet složek
    # images_folders = [images_folders[i] for i in (31,)]  # (10, 11, 12, 13, 19, 33, 37, 38)
    images_folders = [images_folders[0], images_folders[-1]]
    """images_folders = [images_folders[i] for i in range(len(images_folders)) if
                      i not in (10, 11, 12, 13, 19, 33, 37, 38)]"""

    print(f"\nDatum:  {time.strftime('%H:%M, %d.%m. %Y', time.strptime(date, '%H-%M-%S_%d-%m-%Y'))}\n"
          f"\n\033[36mSpuštění programu pro detekci fotek.\n  Verze: {program_version}\n\033[0m"
          f"\nVýpočet proběhne pro složky: \033[34m{images_folders}\033[0m")

    predict_time(folders=images_folders)  # ODHAD CELKOVÉHO ČASU

    if not isinstance(images_folders, list):
        images_folders = [images_folders]

    if not all(not value for value in do_calculations.values()):
        names_csv = check_folder(os.path.join(folder_measurements, "data_csv"),
                                 "Složka s meřeními", "neexistuje", "je prázdná", important_folder=False)

        names_txt = check_folder(os.path.join(folder_measurements, "data_txt"),
                                 "Složka s meřeními", "neexistuje", "je prázdná", important_folder=False)

        if not all(folder in names_csv for folder in images_folders):
            temp_file_list = [file for file in images_folders if not os.path.exists(
                os.path.join(main_image_folder, file, saved_data + ".zip")) or file + ".csv" not in zipfile.ZipFile(
                os.path.join(main_image_folder, file, saved_data + ".zip"), 'r').namelist()]

            temp_file_list = [x for x in temp_file_list if x not in [y for y in names_csv if y in images_folders]]
            temp_file_list = [x for x in temp_file_list if x not in [y for y in names_txt if y in images_folders]]

            if len(temp_file_list) == len(images_folders):
                print(f"\n\033[33;1;21mWARRNING\033[0m\n\tŽádná ze složek {images_folders} nemá přidružená výpočetní "
                      "data CSV nebo TXT.\nJe to v pořádku?")
                while True:
                    if super_speed:
                        ans = "Y"
                    else:
                        ans = askstring("Je to v pořádku?", "Je to v pořádku?\nZadejte Y / N: ")
                        # ans = input("\t\tZadejte Y / N: ")
                    if ans == "Y":
                        print("\n\tZvolena možnost 'Y'\n\t- Výpočet bude pokračovat.")
                        break
                    elif ans == "N":
                        print("\n\tZvolena možnost 'N'\n\t- Zastavení výpočtu.")
                        program_shutdown('\n\033[41;30m➤ Ukončení programu.\033[0m', try_save=False)
                    else:
                        print("\n Zadejte platnou odpověď.")

            elif len(temp_file_list) != 0:
                print(f"\n\033[33;1;21mWARRNING\033[0m"
                      f"\n\tSložky {temp_file_list} nemají přidružená výpočetní data CSV nebo TXT.\nJe to v pořádku?")
                while True:
                    if super_speed:
                        ans = "Y"
                    else:
                        ans = askstring("Je to v pořádku?", "Je to v pořádku?\nZadejte Y / N: ")
                        # ans = input("\t\tZadejte Y / N: ")
                    if ans == "Y":
                        print("\n\tZvolena možnost 'Y'\n\t- Výpočet bude pokračovat.")
                        break
                    elif ans == "N":
                        print("\n\tZvolena možnost 'N'\n\t- Zastavení výpočtu.")
                        program_shutdown('\n\033[41;30m➤ Ukončení programu.\033[0m', try_save=False)
                    else:
                        print("\n Zadejte platnou odpověď.")
            del temp_file_list

        del names_csv, names_txt

    plt.close('all')
    main_counter = 1

    # Zabránění spánku a vypnutí obrazovky
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    # Přepnutí na použití #### TODO optimalizace jader atd
    cv2.setUseOptimized(True)  # Zapnutí optimalizace (může využívat akceleraci)
    cv2.setNumThreads(cv2.getNumThreads())  # Přepnutí na použití CPU počet jader

    variable_names = ['start', 'end', 'all_photos', 'size', 'fine_size', 'points_limit', 'precision', 'scale',
                      'calculations_statuses']

    print(f"\n\033[32mVše je nastaveno.\n\t➤ Spuštení výpočtu:\033[0m")

    # Vytvoření okna
    window_status = tk.Tk()
    window_status.title("Aktuální proces")
    window_status.geometry("370x125")  # Počáteční velikost okna
    window_status.minsize(370, 125)  # Minimální rozměry okna

    # Vytvoření textového pole
    text_entry = tk.Text(window_status, wrap="word", width=40, height=5, padx=10, pady=10)
    text_entry.pack(pady=(10, 0))

    # cyklus mezi složkami - HLAVNÍ CYKLUS
    for current_image_folder in images_folders:
        angle_correction_matrix = None
        photos_times = []

        try:
            update_text_window(f"{current_image_folder}  [ {main_counter} / {len(images_folders)} ]", text_entry)
        except tk.TclError:
            pass

        """plt.close("Status")
        plt.figure(num="Status", figsize=(4.5, 1.5))
        plt.gca().axis('off')
        plt.text(0.5, 0.5, f"{current_image_folder}  [ {main_counter} / {len(images_folders)} ]",
                 fontsize=16, ha='center', va='center')
        plt.tight_layout()
        plt.pause(0.5)
        plt.show(block=False)
        plt.pause(2)"""

        # Spuštění hlavní smyčky
        window_status.update_idletasks()  # Nutné vyvolat aktualizaci, aby se okno zobrazilo
        window_status.update()
        # window_status.deiconify()  # Čekání na zobrazení

        current_image_folder = str(current_image_folder)
        print(f"\033[32;1;21m{'☰' * 50}\033[0m\n\nAktuální výpočet pro: \033[96m{current_image_folder}\033[0m "
              f"\t\t[ {main_counter} / {len(images_folders)} ]")

        saved_data_name = saved_data
        start = start_
        end = end_
        current_folder_path = os.path.join(main_image_folder, current_image_folder)

        """if do_just_correlation:
            saved_data_name = saved_data_name + "_cor"
            if load_calculated_data:
                saved_file_exist = os.path.exists(os.path.join(current_folder_path, saved_data_name + ".zip"))
                if not saved_file_exist:
                    print(f'Pokus o nalezení souboru {saved_data_name + "_cor.zip"}')
                    saved_file_exist = os.path.exists(os.path.join(current_folder_path, saved_data_name + "_cor.zip"))
            else:
                saved_file_exist = False
        else:"""
        saved_file_exist = os.path.exists(os.path.join(current_folder_path, saved_data_name + ".zip"))

        auto_crop = False
        if saved_file_exist and load_calculated_data:
            image_files = get_files_from_zipped_folder(zip_folder_name="image_folder")

            if image_files is None:
                reset_parameters()
                continue

            if end == "all":
                all_photos = True
            else:
                all_photos = False

            try:
                settings_set = [start, end, all_photos, size, fine_size, points_limit, precision, scale,
                                calculations_statuses]
                # Seskupení proměnných aktuálního nastavení do slovníku pro soubor JSON
                if len(variable_names) != len(settings_set):
                    print("\nWARNING:\n\tLength of settings_names and settings_variables are different"
                          f"\n\t\t[settings_names: {len(variable_names) - 1} "
                          f"__ settings_variables: {len(settings_set)}]")

                    print("Chcete ukončit program?")
                    while True:
                        if super_speed:
                            ans = "N"
                        else:
                            ans = askstring("Ukončit program?", "Ukončit program?\nZadejte Y / N: ")
                            # ans = input("\t\tZadejte Y / N: ")
                        if ans == "Y":
                            program_shutdown("\n\tZvolena možnost 'Y'\nUkončení programu.", try_save=False)
                        elif ans == "N":
                            print("\n\tZvolena možnost 'N'\nProgram bude pokračovat.")
                            break
                        else:
                            print("\n Zadejte platnou odpověď.")

                if do_just_correlation:
                    settings_set = settings_set[:2]
                    variable_names = variable_names[:2]
                    settings = {name: value for name, value in zip(variable_names, settings_set)}
                else:
                    settings = {name: value for name, value in zip(variable_names[:-2], settings_set[:-2])}

                """[start, end, main_image_folder, size, fine_size, points_limit, precision, scale,
                 calculations_statuses], dataset_2 = load_data()"""
                (settings, [main_image_folder, current_path_to_photos, scale, _], dataset_2,
                 dataset_3) = load_data()
                main_image_folder, current_path_to_photos = [var.decode('utf-8') for var in
                                                             (main_image_folder, current_path_to_photos)
                                                             if isinstance(var, bytes)]

                if not os.path.isdir(main_image_folder):
                    print(f"Načtená hlavní složka neexituje: [{main_image_folder}]")
                    while not os.path.isdir(main_image_folder):
                        if dynamic_mode:
                            main_image_folder = browse_directory(window_title="Zadejte novou cestu k hlavní složce")
                        else:
                            main_image_folder = input("\tZadejte novou cestu ke složce: ")

                if not os.path.isdir(current_path_to_photos):
                    print(f"Složka s fotkami neexituje: [{current_path_to_photos}]")
                    while not os.path.isdir(current_path_to_photos):
                        if dynamic_mode:
                            current_path_to_photos = browse_directory(
                                window_title="Zadejte novou cestu ke složce s fotkami")
                        else:
                            current_path_to_photos = input("\tZadejte novou cestu ke složce: ")

                """for name, value in zip(variable_names, settings_set):
                    value = settings[name]"""
                for name in settings.keys():
                    globals()[name] = settings.get(name, None)

                # Otevřete ZIP archiv pro čtení
                with zipfile.ZipFile(os.path.join(current_folder_path, saved_data_name + ".zip"), 'r') as zip_file:
                    if [name.startswith("image_folder/") for name in zip_file.namelist()].count(True) <= 1:
                        print('\033[33;1;21mWARRNING\033[0m: Složka "image_folder" není v archivu ZIP nebo je prázdná.')
                        image_files = image_files[start:end]  # TODO ??????????
                    else:
                        pass
                zip_file.close()
                del zip_file
                start = 0

                if not all(key in variable_names for key in settings.keys()):
                    raise MyException(
                        "\n\t\t\t\033[33;1;21mWARRNING\033[0m\n\t\t\tNesoulad uloženého nastavení v souboru JSON."
                        f"\n\t\t\t\tPožadované názvy: {variable_names}"
                        f"\n\t\t\t\tUložené názvy: {settings.keys()}")

                if (calculations_statuses['Correlation']) and (dataset_2['data_correlation'] is not None):
                    if dataset_2['data_correlation'] is not None:
                        correlation_area_points_all = dataset_2['data_correlation']
                        # correlation_area_points_all = [np.array(item) for item in correlation_area_points_all]
                elif (calculations_statuses['Correlation']) != (dataset_2['data_correlation'] is not None):
                    print("\n\033[33;1;21mWARRNING\033[0m"
                          "\n\tNesoulad uložených dat pro data typu: [ 'Correlation' ].")

                if (calculations_statuses['Rough detection']) and (dataset_2['data_rough_detect'] is not None):
                    if dataset_2['data_rough_detect'] is not None:
                        [triangle_vertices_all, triangle_centers_all, triangle_indexes_all,
                         wrong_points_indexes_all, end_marks_all, key_points_all] = dataset_2['data_rough_detect']
                        triangle_points_all = [triangle_vertices_all[i][triangle_indexes_all[i]] for i in
                                               range(len(triangle_vertices_all))]

                elif (calculations_statuses['Rough detection']) != (dataset_2['data_rough_detect'] is not None):
                    print("\n\033[33;1;21mWARRNING\033[0m"
                          "\n\tNesoulad uložených dat pro data typu: [ 'Rough detection' ].")

                if (calculations_statuses['Fine detection']) and (dataset_2['data_fine_detect'] is not None):
                    if dataset_2['data_fine_detect'] is not None:
                        [fine_triangle_points_all, fine_mesh_centers_all] = dataset_2['data_fine_detect']
                elif (calculations_statuses['Fine detection']) != (dataset_2['data_fine_detect'] is not None):
                    print("\n\033[33;1;21mWARRNING\033[0m"
                          "\n\tNesoulad uložených dat pro data typu: [ 'Fine detection' ].")

                if (calculations_statuses['Point detection']) and (dataset_2['data_point_detect'] is not None):
                    if dataset_2['data_point_detect'] is not None:
                        [tracked_points_all, tracked_rotations_all] = dataset_2['data_point_detect']
                elif (calculations_statuses['Point detection']) != (dataset_2['data_point_detect'] is not None):
                    print("\n\033[33;1;21mWARRNING\033[0m"
                          "\n\tNesoulad uložených dat pro data typu: [ 'Point detection' ].")

                if isinstance(dataset_3, dict):
                    for name in dataset_3.keys():
                        globals()[name] = dataset_3.get(name, None)
                else:
                    print("\n\033[33;1;21mWARRNING\033[0m"
                          f"\n\tDataset_3 je špatného typu: [{type(dataset_3)}], správně má být slovník: [dict].")

                del settings, settings_set, dataset_2, dataset_3, name

            except (Exception, MyException) as e:
                dialog(e)

            try:
                set_roi(just_load=True)
            except (Exception, MyException) as e:
                print(f"\nNepovedlo se načíst označené oblasti: {e}")

            print("\nUložená data úspěšně načtena.")

            if calculations_statuses['Rough detection']:
                print(f"\n\tNačteno {len(triangle_centers_all[0])} elemntů.")

            global do_finishing_calculation
            try:
                if do_finishing_calculation:
                    gray1 = load_photo(img_index=0, color_type=photo_type)
                    height, width = gray1.shape[:2]
                    if (((calculations_statuses['Correlation'] == do_calculations['Do Correlation'] and
                          calculations_statuses['Rough detection'] == do_calculations['Do Rough detection'] and
                          calculations_statuses['Fine detection'] == do_calculations['Do Fine detection'] and
                          calculations_statuses['Point detection'] == do_calculations['Do Point detection'])
                         or all(val is False for val in do_calculations.values())
                         or all(val is True for val in calculations_statuses.values()))
                            and not any(val is True for val in recalculate.values())):
                        do_finishing_calculation = False
                        raise MyException(Exception)

                    if save_calculated_data:
                        # Ukládání LOGU
                        with (open(os.path.join(current_folder_path, 'outputs.log'), 'w', encoding='utf-8') as out):
                            # Přesměrování standardního výstupu na objekt Tee
                            sys.stdout = Tee(sys.stdout, out)

                            try:
                                perform_calculations()
                            except Exception as e:
                                print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba výpočtu, Error 1.\n\t\tPOPIS: {e}")
                                out.close()
                                del out
                                reset_parameters()
                                continue

                            out.close()
                            del out

                        try:
                            if try_save_data(zip_name=saved_data_name) is SaveError:
                                raise MyException(Exception)
                            print("\n- Data úspěšně uložena.")
                            if (os.path.exists(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                                    and make_temporary_savings):
                                os.remove(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                                print("\t\t\033[37mDočasné soubory odstraněny\033[0m")
                        except (Exception, MyException) as e:
                            print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba při ukládání dat.\n\t\tPOPIS: {e}")
                    else:
                        try:
                            perform_calculations()
                        except Exception as e:
                            print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba výpočtu, Error 2.\n\t\tPOPIS: {e}")
                            reset_parameters()
                            continue

                        while True:
                            print("\nOpravdu nechcete uložit spočítaná data?")
                            if super_speed and do_finishing_calculation:
                                ans = "Y"
                            elif not do_finishing_calculation:
                                ans = "N"
                            else:
                                ans = askstring("Opravdu nechcete uložit spočítaná data?",
                                                "Opravdu nechcete uložit spočítaná data?\nZadejte Y / N: ")
                                # ans = input("\t\tZadejte Y / N: ")
                            if ans == "Y":
                                print("\n\tZvolena možnost 'Y'\nData budou uložena.")
                                try:
                                    if try_save_data(zip_name=saved_data_name) is SaveError:
                                        raise MyException(Exception)
                                    print("\n- Data úspěšně uložena.")
                                    if (os.path.exists(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                                            and make_temporary_savings):
                                        os.remove(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                                        print("\t\t\033[37mDočasné soubory odstraněny\033[0m")
                                except (MyException, Exception) as e:
                                    print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba při ukládání dat.\n\t\tPOPIS: {e}")
                                break
                            elif ans == "N":
                                print("\n\tZvolena možnost 'N'\nData nebudou uložena.")
                                break
                            else:
                                print("\n Zadejte platnou odpověď.")

                    if (os.path.exists(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                            and make_temporary_savings):
                        os.remove(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                        print("\t\t\033[37mDočasné soubory odstraněny\033[0m")
            except MyException:
                pass

            if scale == 1:
                do_scale(load_photo(img_index=0, color_type=0))

            make_angle_correction()

            """plt.figure(num="Graph of movement of corner to loading bar")  # TODO KONTROLA
            data1 = np.float64([point[0][0] for point in correlation_area_points_all[:]])
            data1 = make_angle_correction(points_to_warp=data1)
            data2 = np.float64([point[1] for point in tracked_points_all[:]])
            data2 = make_angle_correction(points_to_warp=data2)
            data = np.float64([data2[i, 1] - data1[i, 1] for i in range(len(data1))]) * scale
            data -= data[0]

            plt.plot(data, c='dodgerblue', zorder=7)
            plt.scatter(np.arange(len(data)), data, c='darkorange', marker="x", s=25, zorder=6)
            plt.gca().invert_yaxis()
            plt.grid(color="lightgray")
            plt.tight_layout()
            plt.gca().autoscale(True)
            plt.gca().set_aspect('auto', adjustable='box')
            plt.show()"""

            show_results_graph(show_final_image)

            if (len(images_folders) > 1 and  # not super_speed and
                    (calculations_statuses['Correlation'] != do_calculations['Do Correlation'] or
                     calculations_statuses['Rough detection'] != do_calculations['Do Rough detection'] or
                     calculations_statuses['Fine detection'] != do_calculations['Do Fine detection'] or
                     calculations_statuses['Point detection'] != do_calculations['Do Point detection'] or
                     any(val is True for val in recalculate.values()))):
                reset_parameters()
                continue

            if calculations_statuses['Point detection']:
                plot_marked_points(0, show_menu=False, show_arrows=True, save_plot=True, plot_format='jpg',
                                   save_dpi=700, text_size=3, show_marked_points=False)

                # for i in range(len(image_files)):
                plot_point_path(0, show_menu=True, plot_correlation_paths=True, plot_tracked_paths=True, text_size=7)

            for j in [0]:  # TODO KONTROLA
                # import matplotlib.image as mpimg
                img = cv2.cvtColor(load_photo(j, 1), cv2.COLOR_BGR2RGBA)
                # Vytvoření prázdných mask pro obě oblasti
                mask = np.zeros(img.shape[:2], dtype=np.uint8)

                # Vykreslení mnohoúhelníků na maskách
                [cv2.fillPoly(mask, [np.int32(np.round(polygon))], 255) for polygon in triangle_points_all[j]]

                [cv2.rectangle(mask, np.int32(np.round(rectangle[0])), np.int32(np.round(rectangle[1])), 255, -1)
                 for rectangle in correlation_area_points_all[j]]

                # Aplikace mask na obraz
                plt.figure(num="Masked image")
                plt.imshow(cv2.bitwise_and(img, img, mask=mask))

                plt.gcf().set_facecolor((0, 0, 0, 0))
                plt.gca().set_facecolor((0, 0, 0, 0))

                """[plt.gca().add_patch(Polygon(np.array(polygon_coords), edgecolor='b', facecolor='none'))
                 for polygon_coords in triangle_points_all[j]]"""
                plt.tight_layout()
                plt.gca().autoscale(True)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.show()

            plt.figure(num="Graph of loading bar movement")  # TODO KONTROLA
            plt.title("Movement of loading bar")
            data = np.float64([point[0][0] for point in correlation_area_points_all[:]])
            data = make_angle_correction(points_to_warp=data) * scale
            data -= data[0]
            plt.plot(data[:, 0], data[:, 1], c='dodgerblue', zorder=7, label="Path")
            plt.scatter(data[:, 0], data[:, 1], c='darkorange', marker="x", s=25, zorder=6, label="Taken photos")
            plt.gca().invert_yaxis()
            plt.gca().set_xlabel('Horizontal displacement [mm]')
            plt.gca().set_ylabel('Vertival displacement [mm]')
            plt.grid(color="lightgray")
            plt.legend(fancybox=True)
            plt.tight_layout()
            plt.gca().autoscale(True)
            plt.gca().set_aspect('auto', adjustable='box')
            plt.show()

            plot_final_forces(current_image_folder, correlation_area_points_all[:], show_photos=True,
                              interactive_mode=True, fill_area=True)

            if calculations_statuses['Fine detection']:
                # fine_calculation(50)  # fine_calculation(fine_size)  /  fine_calculation2(fine_size)

                show_heat_graph(show_final_image, 0, "Y", fine_triangle_points_all, colorbar_label='[mm]',
                                scaling=scale, centers=fine_mesh_centers_all, make_line_graph=True,
                                save_graph_separately=False, graph_format="jpg", show_correlation_areas=True,
                                image_color_type=0)

            if make_video and calculations_statuses['Fine detection']:
                if not os.path.exists(os.path.join(current_folder_path, "video_output")):
                    os.makedirs(os.path.join(current_folder_path, "video_output"))
                    print(f"Složka {os.path.join(current_folder_path, 'video_output')} byla vytvořena.")

                direction_of_heat_graph = "y"

                print("\nZahájení vytváření videa.")
                original_stdout = sys.stdout  # Uložení původního standardního výstupu
                sys.stdout = open('nul', 'w')  # Nastavení standardního výstupu na nulový objekt

                if direction_of_heat_graph in ("tot", 2):
                    r_values = [(np.linalg.norm(c[:, :] - fine_mesh_centers_all[0][:, :], axis=1)
                                 * scale) for c in fine_mesh_centers_all]
                elif direction_of_heat_graph in ("tot2", 4):
                    r_values = [((c[:, :] - fine_mesh_centers_all[0][:, :]) * scale) for c in
                                fine_mesh_centers_all]
                    r_values = [value[:, 0] + value[:, 1] for value in r_values]
                elif direction_of_heat_graph in ("x", "y", "both", 0, 1, 3):
                    r_values = [((c[:, :] - fine_mesh_centers_all[0][:, :]) * scale) for c in
                                fine_mesh_centers_all]
                    x_data, _, _ = load_forces(current_name=current_image_folder, window_size_average=1)
                    if x_data is not None:
                        x_data = x_data * ((np.linalg.norm(np.array(correlation_area_points_all[-1][0][0]) -
                                                           np.array(correlation_area_points_all[0][0][0])) * scale)
                                           / np.linalg.norm(x_data[-1] - x_data[0]))
                        for vector in r_values:
                            vector[:, 1] += x_data[0]

                max_value = np.max([np.max(vector, axis=0) for vector in r_values], axis=0)
                min_value = np.min([np.min(vector, axis=0) for vector in r_values], axis=0)
                # max_value = np.max(r_values)
                # min_value = np.min(r_values)

                tot_im = len(image_files)

                for im in range(tot_im):
                    show_heat_graph(im, im, direction_of_heat_graph, fine_triangle_points_all, save_graph=True,
                                    heat_values=r_values[im], scaling=scale, colorbar_label='[mm]', save_dpi=400,
                                    make_line_graph=True, max_val=max_value, min_val=min_value, graph_format="jpg",
                                    saved_graph_name=os.path.join("video_output",
                                                                  f"Image_{im:03d}_{image_files[im]}"),
                                    show_correlation_areas=False)
                    plt.cla()
                    plt.close('all')
                    plt.pause(0.5)

                    sys.stdout = original_stdout  # Navrácení původního standardního výstupu
                    print(f"\t\t\033[37mVytvořen graf [{im + 1} / {tot_im + 1}]\033[0m")
                    sys.stdout = open('nul', 'w')  # Nastavení standardního výstupu na nulový objekt

                sys.stdout = original_stdout  # Navrácení původního standardního výstupu
                create_video_from_images(image_folder=os.path.join(current_folder_path, "video_output"),
                                         output_video_path=os.path.join(current_folder_path, "video_output",
                                                                        current_image_folder + ".mp4"),
                                         video_length=30, codec="mp4v", )

        ####################################
        # Do calculation
        else:
            def calculate():
                global image_files, start, end, all_photos, preloaded_images, photos_times

                image_files = sorted(image_files,
                                     key=lambda filename: int(os.path.splitext(filename)[0].split('_')[-1]))

                image_files = image_files[start:end]  # načátání snímků (první je 0) př: "image_files[2:5] od 2 do 5"
                """image_files = [image_files[0], image_files[7], image_files[14], image_files[21],
                               image_files[-1]]"""  # TODO ############ potom změnit počet fotek
                # image_files = [image_files[0], image_files[-1]]

                if preload_photos:
                    preloaded_images = [load_photo(i, photo_type) for i in range(len(image_files))]

                photos_times = [os.path.getmtime(os.path.join(current_path_to_photos, i)) for i in image_files]
                photos_times = [0] + [int(photos_times[i + 1] - photos_times[i]) for i in range(len(photos_times) - 1)]

                if main_counter == 1:
                    print(f"\n\tNastavení:",
                          f"\n\t\tHlavní složka s fotografiemi: ' \033[34m{current_image_folder}\033[0m '",
                          f"\n\t\tCesta k aktuální složce: ' \033[35m{current_path_to_photos}\033[0m '",
                          f"\n\t\tVelikost elementů: {size}",
                          f"\n\t\tVelikost pod-elementů: {fine_size}",
                          f"\n\t\tMinimální počet bodů: {points_limit}",
                          f"\n\t\tPřesnost nalezených bodů: {precision}\n")

                perform_calculations()

                print(f"\nDokončení výpočtu {main_counter}. složky ' {current_image_folder} '"
                      f"\t\t[ {main_counter} / {len(images_folders)} ]")

            """# Kontrola zda aktuální složka existuje a není prázdná
            if os.path.isdir(current_folder_path):
                if not os.listdir(current_folder_path):
                    program_shutdown(f"\nERROR:\n\tSložka měření s fotkami: ' {current_folder_path} ' je prázdná."
                          f"\n\t➤ Ukončení programu.")
            else:
                program_shutdown(f"\nERROR:\n\tSložka měření s fotkami: ' {current_folder_path} ' neexistuje."
                      f"\n\t➤ Ukončení programu.")"""

            """# Vytvoření cesty k cílovým fotografiím
            current_path_to_photos = os.path.join(current_folder_path, source_image_type[1])
            # Kontrola zda aktuální složka existuje a není prázdná a případně změna stránky
            if not os.path.isdir(current_path_to_photos):
                current_path_to_photos = os.path.join(current_folder_path, source_image_type[0])
                if not os.path.isdir(current_path_to_photos):
                    print(f"\n\033[33;1;21mWARRNING\033[0m\n\t"
                          f"Aktuální složka {current_folder_path} neobsahuje fotografie v samostatných složkách.\n")
                    reset_parameters()
                    continue"""

            # Vytvoření cesty k cílovým fotografiím
            current_path_to_photos = os.path.join(current_folder_path, source_image_type[1])
            if not any(f.endswith(photos_types) for f in os.listdir(current_path_to_photos)):
                print(f"\nAktuální složka: ' \033[35m{current_path_to_photos}\033[0m ' \033[91mje prázdná.\033[0m"
                      f"\n\t\tKontrola další složky.")
                current_path_to_photos = os.path.join(current_folder_path, source_image_type[0])
                print(f"Změna aktuální složky na: ' \033[35m{current_path_to_photos}\033[0m '")
                if os.path.isdir(current_path_to_photos):
                    if not any(f.endswith(photos_types) for f in os.listdir(current_path_to_photos)):
                        if len(images_folders) <= 1:  # or current_image_folder == images_folders[-1]:
                            program_shutdown(f"\n\033[31;1;21mERROR\033[0m"
                                             f"\n\tAktuální složka: [\033[35m{current_path_to_photos}\033[0m] "
                                             f"\033[91mje prázdná.\033[0m\n\t\033[41;30m➤ Ukončení programu.\033[0m",
                                             try_save=False)
                        else:
                            print(f"\n\033[33;1;21mWARRNING\033[0m"
                                  f"\n\tAktuální složka: [\033[35m{current_path_to_photos}\033[0m] "
                                  f"\033[91mje prázdná.\033[0m\n\t- Tato složka bude přeskočena.")
                            reset_parameters()
                            continue
                    print(f"\t\tNová složka je v pořádku.\n")
                else:
                    if len(images_folders) <= 1:  # or current_image_folder == images_folders[-1]:
                        program_shutdown(f"\n\nŽádná ze složek nevyhovuje.\n\t\033[41;30m➤ Ukončení programu.\033[0m",
                                         try_save=False)
                    else:
                        print(f"\n\033[33;1;21mWARRNING\033[0m"
                              f"\n\tAktuální složka: [\033[35m{current_path_to_photos}\033[0m] je prázdná."
                              f"\n\t- Tato složka bude přeskočena.")

            # Získání seznamu souborů ve složce
            file_list = os.listdir(current_path_to_photos)

            if not (saved_file_exist and load_calculated_data):
                # Omezeni počtu snímků
                if dynamic_mode:
                    image_files = browse_files(window_title="Vyberte fotografie")
                    start, end, all_photos = 0, len(image_files), False
                else:
                    image_files = get_photos_from_folder(current_path_to_photos, file_list,
                                                         (".jpg", ".jpeg", ".jpe", ".JPG", ".jp2", ".png", ".bmp",
                                                          ".dib", ".webp", ".avif", ".pbm", ".pgm", ".ppm", ".pxm",
                                                          ".pnm", ".pfm", ".sr", ".ras", ".tiff", ".tif", ".exr",
                                                          ".hdr", ".pic"))
                    if not image_files:
                        print(f"\n\033[31;1;21mERROR\033[0m\n\tSložka neobsahuje fotografie.")
                        reset_parameters()
                        continue
                    elif image_files is None:
                        reset_parameters()
                        continue

                    if end == "all":
                        end = len(image_files)
                        all_photos = True
                    else:
                        all_photos = False

            if save_calculated_data or second_callout:
                # Ukládání LOGU
                with (open(os.path.join(current_folder_path, 'outputs.log'), 'w', encoding='utf-8') as out):
                    # Přesměrování standardního výstupu na objekt Tee
                    sys.stdout = Tee(sys.stdout, out)

                    try:
                        calculate()
                    except Exception as e:
                        print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba výpočtu, Error 3.\n\t\tPOPIS: {e}")
                        out.close()
                        reset_parameters()
                        continue

                    out.close()
                    del out
                    try:
                        if try_save_data(zip_name=saved_data_name) is SaveError:
                            raise MyException(Exception)

                        print("\n- Data úspěšně uložena.")

                        if (os.path.exists(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                                and make_temporary_savings):
                            os.remove(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                            print("\t\t\033[37mDočasné soubory odstraněny\033[0m")

                    except (Exception, MyException) as e:
                        print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba při ukládání dat.\n\t\tPOPIS: {e}")

                    scale, second_callout, saved_data_name = 1, False, saved_data
            else:
                calculate()
                try:
                    calculate()
                except Exception as e:
                    print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba výpočtu, Error 4.\n\t\tPOPIS: {e}")
                    reset_parameters()
                    continue
                while True:
                    print("\nOpravdu nechcete uložit spočítaná data?")
                    if super_speed:
                        ans = "Y"
                    else:
                        ans = askstring("Opravdu nechcete uložit spočítaná data?",
                                        "Opravdu nechcete uložit spočítaná data?\nZadejte Y / N: ")
                        # ans = input("\t\tZadejte Y / N: ")
                    if ans == "Y":
                        print("\n\tZvolena možnost 'Y'\nData budou uložena.")
                        try:
                            if try_save_data(zip_name=saved_data_name) is SaveError:
                                raise MyException(Exception)

                            print("\n- Data úspěšně uložena.")

                            if (os.path.exists(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                                    and make_temporary_savings):
                                os.remove(os.path.join(current_folder_path, f"{saved_data_name}_temp.zip"))
                                print("\t\t\033[37mDočasné soubory odstraněny\033[0m")

                        except (Exception, MyException) as e:
                            print(f"\n\033[31;1;21mERROR\033[0m\n\tChyba při ukládání dat.\n\t\tPOPIS: {e}")
                        break
                    elif ans == "N":
                        print("\n\tZvolena možnost 'N'\nData nebudou uložena.")
                        break
                    else:
                        print("\n Zadejte platnou odpověď.")

        reset_parameters()

    if send_final_message:
        current_time = time.time()
        time_difference = (current_time - time.mktime(time.strptime(date, "%H-%M-%S_%d-%m-%Y")))

        message = (f"Detection program has finished.\nProgram version: {program_version}\n",
                   f"\tStart of program: "
                   f"[{time.strftime('%H:%M:%S, %d.%m. %Y', time.strptime(date, '%H-%M-%S_%d-%m-%Y'))}]  --  "
                   f"End of program: [{time.strftime('%H:%M:%S, %d.%m. %Y', time.localtime(current_time))}]\n",
                   (f"\tTotal computation time : {np.int32(time_difference // 3600)} hours, "
                    f"{np.int32((time_difference % 3600) // 60)} minutes, {np.int32(time_difference % 60)} seconds"),
                   f"\n{'=' * 80}\n",
                   "\n\nCalculation:\n",
                   f"\tTotal finished calculations: {main_counter - 1}\n",
                   f"\t\t- Calculated files:  {images_folders}\n\n",
                   f"\t\t- Done types of calculations:",
                   f"\n\t\t\t\t\tCorrelation:{13 * ' '}{do_calculations['Do Correlation']}",
                   f"\n\t\t\t\t\tRough detection:{5 * ' '}{do_calculations['Do Point detection']}",
                   f"\n\t\t\t\t\tFine detection:{8 * ' '}{do_calculations['Do Fine detection']}",
                   f"\n\t\t\t\t\tPoint detection:{7 * ' '}{do_calculations['Do Point detection']}\n\n",
                   f"\t\t{'Loading preset points was set to:':<42}{load_set_points}\n",
                   f"\t\t{'Mark points automatically was set to:':<40}{do_auto_mark}\n",
                   f"\t\t{'Marking points by hand was set to:':<40}{mark_points_by_hand}\n",
                   "\n\t\t- Was done just correlation of areas.\n" if do_just_correlation else (
                       (f"\n\t\t- Normal calculation:\n\t\t\tSize of elements: {size}"
                        f"\n\t\t\tPrecision limit: {precision}\n\t\t\tLimit of points: {points_limit}\n"),
                       (f"\n\t\t- Fine calculation:\n\t\t\tSize of fine elements: {fine_size}\n"
                        if do_calculations['Do Fine detection'] else "\n")),
                   f"\n\tData was loaded from file: {saved_data_name}.zip\n\n"
                   if saved_file_exist and load_calculated_data else "\n",
                   f"\tData was saved to file: {saved_data_name}.zip\n"
                   if save_calculated_data and not (saved_file_exist and load_calculated_data)
                   else "\tData was not saved.\n")

        def flatten_tuple(input_tuple):
            return [item for sublist in input_tuple for item in
                    (flatten_tuple(sublist) if isinstance(sublist, (tuple, list)) else
                     (str(list(sublist.reshape(-1, ))) if isinstance(sublist, np.ndarray) else
                      (str(sublist) if isinstance(sublist, (int, float)) else (sublist,))))]

        message = flatten_tuple(message)
        # Spojení hodnot z lis/tuple do jednoho stringu
        message = " ".join(message)

        send_out_message(send_sms=True, send_mail=True, sms_body="Program is finished.",
                         mail_body=message, name_of_sender="Detection Program", time_stamp=current_time)

    # Navrácení režimu spánku
    plt.close('all')
    # ukončení hlavní smyčky
    try:
        window_status.destroy()
    except tk.TclError:
        pass
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print("\nUkončení programu.")


def reset_parameters():
    global scale, second_callout, saved_data_name, calculations_statuses, main_counter, start, end

    start = start_
    end = end_
    main_counter += 1
    plt.close('all')
    scale, second_callout, saved_data_name = float(1), False, saved_data
    calculations_statuses = {key: False for key in calculations_statuses}

    """scale, second_callout, saved_data_name = 1, False, saved_data
    calculations_statuses = {key: False for key in calculations_statuses}
    # Odstraňování proměnných
    for name in ('keypoints1_sift', 'descriptors1_sift', 'fine_triangle_points_all', 'fine_mesh_centers_all',
                 'points_pos', 'points_neg', 'points_cor', 'points_max', 'saved_file_exist', 'wrong_points_indexes_all',
                 'key_points_all', 'end_marks_all', 'triangle_vertices_all', 'triangle_centers_all',
                 'triangle_indexes_all', 'triangle_points_all', 'correlation_area_points_all', 'settings'):
        if name in globals():
            del globals()[name]"""

    return


if __name__ == '__main__':
    global start, end, folder_measurement, image_files, current_path_to_photos, settings, variable_names, main_counter
    global gray1, gray2, width, height, auto_crop, saved_data_name, all_photos, preloaded_images
    global triangle_vertices_all, triangle_centers_all, triangle_indexes_all, triangle_points_all, \
        correlation_area_points_all, wrong_points_indexes_all, key_points_all, end_marks_all, tracked_points_all, \
        tracked_rotations_all, angle_correction_matrix, photos_times
    global points_pos, points_neg, points_cor, points_max, points_track, photo_size
    global saved_file_exist, current_folder_path, current_image_folder
    global fine_triangle_points_all, fine_mesh_centers_all
    global keypoints1_sift, descriptors1_sift

    # Spuštění programu
    print("\n\033[34;21mSpouštění programu.\033[0m\n")

    from matplotlib.widgets import CheckButtons  # , TextBox
    from matplotlib.widgets import PolygonSelector, RectangleSelector, EllipseSelector
    from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, FancyArrowPatch
    import matplotlib.patheffects as pe

    import tkinter as tk
    from tkinter import ttk
    from tkinter import Entry
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    from tkinter.simpledialog import askfloat, askinteger, askstring

    from pyzbar.pyzbar import decode as qr_detect

    # import keyboard

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    #                                                                                                                  #
    # ########################### ##########              SETTINGS              ########## ########################### #
    #                                                                                                                  #
    ####################################################################################################################

    block_graphs = True  # False = kód poběží dál / True = kód se zastaví

    dynamic_mode = False

    send_final_message = False

    load_set_points = True
    do_auto_mark = False
    mark_points_by_hand = True

    do_calculations = {'Do Correlation': True,
                       'Do Rough detection': True,
                       'Do Fine detection': False,
                       'Do Point detection': True}

    main_image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'

    folder_measurements = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data'

    data_type = "M01"

    templates_path = folder_measurements + r'\templates\templates_H01'

    source_image_type = ['original', 'modified']

    saved_data = 'pokus'  # data_export_new
    save_calculated_data = False
    load_calculated_data = False
    do_finishing_calculation = False
    make_temporary_savings = False

    make_video = False

    start_, end_ = 1, "all"

    size = 200  # !=_ 135 _=!,   250 - pro hexagony , 100, 85 - min,   (40)
    fine_size = 20  # np.int32(size * 0.1)

    points_limit = 1200
    precision = 0.65

    show_final_image = -1  # Kterou fotografii vykreslit

    program_version = 'v0.8.50'

    preload_photos = False

    super_speed = True

    photo_type = 0

    set_n_features = 0
    set_n_octave_layers = 3
    set_contrast_threshold = 0.08
    set_edge_threshold = 15
    set_sigma = 1.6

    recalculate = {'Re Correlation': False,
                   'Re Rough detection': False,
                   'Re Fine detection': False,
                   'Re Point detection': False}

    #                                                                                                                  #
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    #                                                                                                                  #

    date = get_current_date()

    ctu_color = {"CTU_blue": '#0065BD',
                 "CTU_skyblue": '#6AADE4',
                 "CTU_petrol": '#156570',
                 "CTU_seagreen": '#00B2A9',
                 "CTU_firebrick": '#C60C30',
                 "CTU_orange": '#E05206',
                 "CTU_wine": '#981F40',
                 "CTU_pink": '#F04D98',
                 "CTU_honey": '#F0AB00',
                 "CTU_green": '#A2AD00',
                 "CTU_gray": '#9B9B9B', }

    scale, second_callout = float(1), False
    calculations_statuses = {'Correlation': False, 'Rough detection': False,
                             'Fine detection': False, 'Point detection': False}

    # list(do_calculations.values())[0] is True and all(value is False for value in list(do_calculations.values())[1:]):
    if do_calculations["Do Correlation"] and all(
            not value for key, value in do_calculations.items() if key != "Do Correlation"):
        do_just_correlation = True
    else:
        do_just_correlation = False

    if super_speed:
        block_graphs = False
    if save_calculated_data or load_calculated_data or super_speed:
        import h5py
    # if not do_just_correlation:
    import pygmsh
    # if do_calculations['Do Fine detection']:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # if do_calculations['Do Correlation']:
    import json
    # if not all(not value for value in do_calculations.values()):
    import zipfile

    if do_calculations['Do Point detection']:
        from matplotlib.path import Path

    if dynamic_mode:
        from tkinter import filedialog

    if not save_calculated_data:
        make_temporary_savings = False

    # block_graphs = True  # TODO KONTROLA

    if dynamic_mode:
        main_image_folder = browse_directory(window_title="Vyberte hlavní složku s fotografiemi")
        folder_measurements = browse_directory(window_title="Vyberte hlavní složku s meřeními")

    saved_data = saved_data.replace(".zip", "").replace(".", "_")

    """photos_types = (".jpg", ".jpeg", ".jpe", ".JPG", ".jp2", ".png", ".bmp",
                    ".dib", ".webp", ".avif", ".pbm", ".pgm", ".ppm", ".pxm",
                    ".pnm", ".pfm", ".sr", ".ras", ".tiff", ".tif", ".exr",
                    ".hdr", ".pic")"""

    photos_types = (".jpg", ".jpeg", ".JPG", ".png", ".tiff", ".tif")

    main()

    # ━━━━━━━━━

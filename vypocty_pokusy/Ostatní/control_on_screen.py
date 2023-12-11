import time
import pyautogui

time.sleep(10)
print("Start")

# Identifikujte relativní pozici textového pole nebo jiného prvku
# (pozice může být potřeba upravit podle konkrétní aplikace a rozložení okna)
relative_text_field_position = (100, 100)

# Identifikujte relativní pozici tlačítka nebo jiného prvku, na které chcete kliknout
# (pozice může být potřeba upravit podle konkrétní aplikace a rozložení okna)
relative_button_position = (500, 40)

# Získání aktuální pozice myši
current_mouse_x, current_mouse_y = pyautogui.position()

# Výpočet relativních pozic
text_field_position = (current_mouse_x + relative_text_field_position[0], current_mouse_y + relative_text_field_position[1])
button_position = (current_mouse_x + relative_button_position[0], current_mouse_y + relative_button_position[1])

# Simulace kliknutí do textového pole
pyautogui.click(*text_field_position)

# Pauza pro počkání na fokus na textovém poli (čas může být potřeba upravit)
time.sleep(1)

# Simulace psaní do textového pole
pyautogui.typewrite("Hello, this is automated typing!")

# Simulace kliknutí na tlačítko
pyautogui.click(*button_position)

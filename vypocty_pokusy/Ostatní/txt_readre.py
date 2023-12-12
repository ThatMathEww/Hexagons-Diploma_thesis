import pyautogui
import time


time.sleep(10)

# Přejde na specifické souřadnice, klikne a vloží text
pyautogui.click(*(940, 300))
# pyautogui.hotkey('ctrl', 'a')
pyautogui.typewrite("P")

# Přejde na další souřadnice a klikne
# pyautogui.click(*(1180, 230))
# Simulace stisknutí klávesy Tab + Shift
pyautogui.hotkey('shift', 'tab')
pyautogui.press('enter')

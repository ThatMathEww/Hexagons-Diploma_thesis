import numpy as np
import time
import matplotlib.pyplot as plt


def create_progress_bar(current_number, total_number, bar_length=20):
    progress = min(current_number / total_number, 1.0)
    num_bars = np.int32(np.round(bar_length * progress))
    bar = "▰" * num_bars + "▱" * (bar_length - num_bars)
    percentage = f"{np.int32(np.round(progress * 100 / 5) * 5):3d}%"  # rounded progress to 5
    text = f"{bar}  {percentage}"
    print(f"\r{text}", end="")


# Příklad použití:
total = 236
length = 20
print("𝐍𝐨𝐰 𝐥𝐨𝐚𝐝𝐢𝐧𝐠...")
for current in range(0, total + 1, 10):
    create_progress_bar(current, total, length)
    plt.pause(0.1)
    # time.sleep(0.5)
_ = "▰" * length
_ = f"{_}  100%"
print(f"\r{_}", end="")
del _
print()  # Zde závěrečný přechod na nový řádek, aby se oddělil interaktivní tisk od dalšího výstupu.
print("⋘ 𝑃𝑙𝑒𝑎𝑠𝑒 𝑤𝑎𝑖𝑡...⋙")
print("ᴄᴏᴍᴘʟᴇᴛᴇ!\n")

for i in range(10):
    a = i % 3
    dots = ["▁ . .", ". ▁ .", ". . ▁"]
    t = f"⋘ loading data {dots[a]} ⋙"
    print(f"\r{t}", end="")
    plt.pause(0.5)
print(f"\r⋘ Data loaded ⋙", end="")
print()

# ▓ ░ , █ ▁ , ■ □ , ● ○, [■■■□□□□□□□] 30%, ▒▒▒▒▒▒▒▒▒▒ 100% ᴄᴏᴍᴘʟᴇᴛᴇ!  𝐍𝐨𝐰 𝐥𝐨𝐚𝐝𝐢𝐧𝐠... ⋘ 𝑃𝑙𝑒𝑎𝑠𝑒 𝑤𝑎𝑖𝑡...⋙
# ⋘ 𝑙𝑜𝑎𝑑𝑖𝑛𝑔 𝑑𝑎𝑡𝑎...⋙ ⋘ ᴛʀʏ ʟᴀᴛᴇʀ... ⋙, ▰▱, ⚫⚪, ⬛⬜, ◼▭, ▮▯
# ##### " https://changaco.oy.lc/unicode-progress-bars/ " , " https://copy-paste.net/en/loading-bar.php "
# ##### " https://www.textfacescopy.com/loading-symbol.html "



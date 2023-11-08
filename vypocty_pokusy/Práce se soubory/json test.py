import json
import numpy as np
import sys

if 'json' in sys.modules:
    # Knihovna 'json' již byla načtena
    print("Knihovna 'json' byla již načtena.")
else:
    # Knihovna 'json' ještě nebyla načtena
    print("Knihovna 'json' dosud nebyla načtena.")

data = np.array(((100, 200), (50, 30)))

with open('j.json', 'w') as file:
    json.dump(data.tolist(), file, indent="\t")
file.close()

with open('j.json', 'r') as file:
    loaded = json.load(file)
file.close()
loaded = np.array(loaded)
print(loaded)

print('Done')

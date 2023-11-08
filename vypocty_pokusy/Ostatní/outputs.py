import numpy as np

data = [
    {"comment": "První výpočet", "value": 42.0},
    {"comment": "Druhý výpočet", "value": 15.3},
    {"comment": "Třetí výpočet", "value": None}
]

output_lines = [f"Komentář: {item['comment']}\n\tHodnota: {item['value']}\n\n" for item in data]

with open('output.txt', 'w') as txt_file:
    txt_file.writelines(output_lines)
    txt_file.close()

with open('output.txt', 'r') as txt_file:
    lines = txt_file.readlines()

lines = [line.strip() for line in lines]
comment_lines = lines[::3]
value_lines = lines[1::3]

comments = [None if comment == "Komentář:" else comment.split("Komentář: ")[1] for comment in comment_lines]
values = [line.split("Hodnota: ")[1] for line in value_lines]


def process_value(val):
    try:
        val = float(val)
        return int(val) if val.is_integer() else val
    except (ValueError, TypeError):
        val = None
    return val


# Vektorizovaná úprava členů s ohledem na hodnoty None
process_value_vectorized = np.vectorize(process_value, otypes=[object])
values = process_value_vectorized(values)

data = [{"comment": comment, "value": value} for comment, value in zip(comments, values)]

for item in data:
    comment = item.get('comment')
    value = item.get('value')
    result_str = (f'{f"Komentář:  {comment}" if comment else "":<30}'
                  f'{", " if value and comment else "  "}'
                  f'{f" Hodnota:  {value}" if value else ""}')
    print(result_str)

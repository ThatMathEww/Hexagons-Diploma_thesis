import timeit
import random


# Testovací funkce pro metodu "append"
def method_append(base_list, extension_list):
    list_new = base_list.copy()
    for _ in range(1000):
        list_new.append(extension_list)
    return list_new


# Testovací funkce pro metodu "extend"
def method_extend(base_list, extension_list):
    list_new = base_list.copy()
    for _ in range(1000):
        list_new.extend(extension_list)
    return list_new


# Testovací funkce pro metodu "slicing =+"
def method_slicing(base_list, extension_list):
    list_new = base_list.copy()
    for _ in range(1000):
        list_new += extension_list
    return list_new


new_elements = [4, 5, 6]

my_list1 = []
for _ in range(3):
    my_list1.append(new_elements)  # Přidá celý seznam 'new_elements' jako jeden prvek
print(my_list1)  # Výstup: [1, 2, 3, [4, 5, 6]]

my_list2 = []
for _ in range(3):
    my_list2.extend(new_elements)  # Přidá prvky z 'new_elements' na konec seznamu
print(my_list2)  # Výstup: [1, 2, 3, 4, 5, 6]

# Počet opakování pro měření doby běhu
num_iterations = 100

empty_list = []
# Vytvoření náhodného listu délky 100 s náhodnými čísly
# random_list = [random.randint(1, 1000) for _ in range(100)]

# Nebo pro náhodné hodnoty s desetinnými čísly
random_list = [random.uniform(0, 1) for _ in range(10000)]

# Testování a měření času pro každou metodu
append_time = timeit.timeit(lambda: method_append(empty_list, random_list), number=num_iterations)
extend_time = timeit.timeit(lambda: method_extend(empty_list, random_list), number=num_iterations)
slicing_time = timeit.timeit(lambda: method_slicing(empty_list, random_list), number=num_iterations)

m1 = method_append(empty_list, random_list)
m2 = method_extend(empty_list, random_list)
m3 = method_slicing(empty_list, random_list)

print("\nHotovo.")

if m1 == m2 == m3:
    print("Jsou stejné všechny.\n\n")
elif m2 == m3:
    print("1 a 2 jsou stejné.\n\n")

# Výsledky
print(f"Metoda 'append' : {round(append_time, 3)} sekund pro {num_iterations} opakování")
print(f"Metoda 'extend' : {round(extend_time, 3)} sekund pro {num_iterations} opakování")
print(f"Metoda 'slicing': {round(slicing_time, 3)} sekund pro {num_iterations} opakování")

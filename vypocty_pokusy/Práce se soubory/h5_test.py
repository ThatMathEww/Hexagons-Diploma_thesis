import h5py


# Vytvořte slovník
my_dict = {"a": False, "baa": True}

# Uložit proměnnou do hlavní skupiny
main_variable = 42

# Otevřete soubor HDF5 pro zápis
file = h5py.File('mojefile.h5', 'w')

# Vytvořte hlavní skupinu (main group)
main_group = file.create_group('main_group')

# Uložte proměnnou do hlavní skupiny
main_group.attrs['main_variable'] = main_variable

# Vytvořte subskupinu (subgroup) v hlavní skupině
sub_group = main_group.create_group('sub_group')

# Uložte slovník do subskupiny jako atribut
for key, value in my_dict.items():
    sub_group.attrs[key] = value


# Uzavřete soubor HDF5
file.close()

# Otevřete soubor HDF5 pro čtení
with h5py.File('mojefile.h5', 'r') as file:

    # Získat hodnotu proměnné z hlavní skupiny
    main_variable = file['main_group'].attrs['main_variable']

    # Získejte subskupinu (subgroup) z hlavní skupiny (main group)
    sub_group = file['main_group/sub_group']

    main_group = file['main_group']

    # Načtěte slovník z atributů subskupiny
    my_dict = {key: bool(value) for key, value in sub_group.attrs.items()}

    # Získáme seznam skupin ve souboru
    group_names = list(sub_group.keys())

    # Vytiskneme seznam skupin ve souboru
    print("Seznam skupin ve souboru:")
    for group_name in group_names:
        print(group_name)

# Uzavřete soubor HDF5
file.close()
print(my_dict)

print("\nDone")

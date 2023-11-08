import yaml

data = {"person":
            {"name": "John", "age": 30, "address":
                {"street": "123 Main St", "city": "Springfield"
                 }
             }
        }

with open("data.yml", "w") as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)

with open("data.yml", "r") as yaml_file:
    loaded_data = yaml.safe_load(yaml_file)


print(data == loaded_data)
print(loaded_data)

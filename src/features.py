features = [
    "model_year",
    "brand_name",
    "model_name",
    "variant_name",

    "km_driven",
    "fuel_type",
    "transmission_type",
    "owner_type",
    "seller_type_new",

    "state",
    "city_name_new",

    "car_segment",
    "Engine Type",
    "Displacement",
    "Max Power",
    "Max Torque",
    "No of Cylinder",
    "Values per Cylinder",
    "Value Configuration",
    "Turbo Charger",
    "Super Charger",

    "Length",
    "Width",
    "Height",
    "Wheel Base",
    "Kerb Weight",

    "Gear Box",
    "Drive Type",
    "seating_capacity_new",
    "Steering Type",

    "Front Brake Type",
    "Rear Brake Type",
    "Top Speed",
    "Acceleration",
    "Tyre Type",
    "No Door Numbers",
    "Cargo Volumn",

    "mileage_new",
    "exterior_color",

    "top_features_count",
    "comfort_features_count",
    "interior_features_count",
    "exterior_features_count",
    "safety_features_count",

    "Fuel Suppy System",
    "Alloy Wheel Size",

    "imgCount",
    "threesixty",
]

# Limpiada
categorical_cols = [
    "brand_name",
    "model_name",
    "variant_name",
    "fuel_type",
    "transmission_type",
    "owner_type",
    "seller_type_new",
    "state",
    "city_name_new",
    "car_segment",
    "Engine Type",
    "Value Configuration",
    "Gear Box",
    "Drive Type",
    "Steering Type",
    "Front Brake Type",
    "Rear Brake Type",
    "Tyre Type",
    "exterior_color",
    "Fuel Suppy System",
]

# Limpiada
boolean_text_cols = [
    "Turbo Charger",
    "Super Charger",
]

# No hace falta limpiar, ya que son booleanos
boolean_cols = [
    "threesixty",
]

# No hace falta limpiar, ya que son numericos
numeric_cols = [
    "model_year",
    "km_driven",
    "Displacement",
    "No of Cylinder",
    "Values per Cylinder",
    "seating_capacity_new",
    "No Door Numbers",
    "imgCount",
]

# Limpiada
mm_cols = [
    "Length",
    "Width",
    "Height",
    "Wheel Base",
]

# Limpiada
kg_cols = [
    "Kerb Weight",
]

# Limpiada
list_cols = [
    "top_features",
    "comfort_features",
    "interior_features",
    "exterior_features",
    "safety_features",
]

# No es necesario limpiar, ya que son numericos
list_cols_count = [
    "top_features_count",
    "comfort_features_count",
    "interior_features_count",
    "exterior_features_count",
    "safety_features_count",
]

# Limpiada
convert_to_number = [
    "Max Power",                 # bhp
    "Max Torque",                # Nm
    "Top Speed",                 # kmph
    "Acceleration",              # segundos
    "Cargo Volumn",              # litros
    "Alloy Wheel Size",          # pulgadas
]

mileage_cols = [
    "mileage_new",
]
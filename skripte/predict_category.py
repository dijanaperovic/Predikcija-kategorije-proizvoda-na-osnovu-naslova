import pandas as pd
import joblib

import os
os.chdir(r"C:\Users\Dijana\Desktop\IT Academy\Machine Learning\Predikcija-kategorije-proizvoda-na-osnovu-naslova")
print("Trenutni radni direktorijum:", os.getcwd())

model = joblib.load("category_prediction_model.pkl") 
print("Model je uspešno učitan!\n")

# Lista brendova za feature 'has_brand'
brands_list = [
    'iPhone', 'Samsung', 'Sony', 'Philips', 'Bosch',
    'LG', 'Panasonic', 'Apple', 'Dell', 'HP', 'Lenovo', 'Whirlpool'
]

while True:
    title = input("Unesi naziv proizvoda (ili 'exit' za kraj): ")
    if title.lower() == "exit":
        print("Izlazak iz programa...")
        break

    # Feature engineering za unos korisnika
    product_name_len = len(title)
    number_in_name = int(any(ch.isdigit() for ch in title))
    has_brand = int(any(brand.lower() in title.lower() for brand in brands_list))

    # Kreiranje DataFrame-a za predikciju
    user_input = pd.DataFrame([{
        "product_title": title,
        "product_name_len": product_name_len,
        "number_in_name": number_in_name,
        "has_brand": has_brand
    }])

    # Predikcija kategorije
    prediction = model.predict(user_input)[0]
    print(f"Predviđena kategorija: {prediction}\n{'-'*40}")

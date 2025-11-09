# Predikcija-kategorije-proizvoda-na-osnovu-naslova

## Opis projekta

Ovaj projekat koristi mašinsko učenje kako bi predvideo kategoriju proizvoda na osnovu naziva proizvoda. Model je treniran koristeći LinearSVC i dodatne inženjerske karakteristike kao što su: dužina naziva proizvoda, prisustvo brojeva u nazivu i prepoznati brendovi.

Dataset sadrži proizvode iz deset kategorija:
mobile phones, tvs, cpus, digital cameras, microwaves, dishwashers, washing machines, freezers, fridge freezers, fridges.

## Struktura projekta

**data/products.csv** – dataset sa nazivima proizvoda i njihovim kategorijama.

**category_prediction_model.pkl** – sačuvani trenirani model.

**notebook.ipynb** – Colab notebook sa kodom za treniranje, testiranje i interaktivnu predikciju.

## Uputstvo za korišćenje modela

1. **Pristup modelu**  
Model je sačuvan na Google Drive-u i dostupan za upotrebu u Google Colab-u ili lokalnom Python okruženju. Putanja do modela je: `/content/drive/MyDrive/models/category_prediction_model.pkl`.

2. **Pokretanje interaktivnog testiranja**  
Otvorite Colab svesku koja učitava model. Nakon učitavanja, možete unositi nazive proizvoda za koje želite predikciju kategorije. Model će za svaki naziv proizvoda prikazati predviđenu kategoriju.

3. **Unos proizvoda**  
Unesite naziv proizvoda koji želite da klasifikujete. Ako naziv sadrži broj ili poznati brend, model će to automatski koristiti kao dodatnu informaciju. Lista prepoznatih brendova uključuje: iPhone, Samsung, Sony, Philips, Bosch, LG, Panasonic, Apple, Dell, HP, Lenovo, Whirlpool. Kada završite testiranje, unesite `exit` da biste izašli iz interaktivnog režima.

4. **Napomene**  
Model je treniran na deset kategorija proizvoda: mobile phones, tvs, cpus, digital cameras, microwaves, dishwashers, washing machines, freezers, fridge freezers, fridges. Model koristi pored naziva proizvoda i dodatne karakteristike (dužina naziva, prisustvo brojeva i prisustvo brenda) kako bi poboljšao tačnost predikcije. Za nove proizvode ili brendove koji nisu u listi, preporučuje se ažuriranje modela ili liste brendova radi preciznije predikcije.

import pandas as pd

import os
os.chdir(r"C:\Users\Dijana\Desktop\IT Academy\Machine Learning\Predikcija-kategorije-proizvoda-na-osnovu-naslova")
print("Trenutni radni direktorijum:", os.getcwd())

df=pd.read_csv("data/products.csv")
df = df.dropna()
print(df.shape)
print(df.isna().sum())
#Standardizovanje naziva kolona za lepši prikaz i lakše snalaženje
df.columns = df.columns.str.lower().str.strip().str.replace(r' ', '_', regex = True)
df['category_label'] = df['category_label'].astype(str).str.strip().str.lower()
category_map = {
    'mobile phone': 'mobile phones',
    'cpu': 'cpus',
    'fridge': 'fridges'
}
df['category_label'] = df['category_label'].replace(category_map) 

#Nove karakteristike
df['product_name_len'] = df['product_title'].astype(str).str.len()
df['number_in_name'] = df['product_title'].apply(lambda x: int(any(ch.isdigit() for ch in x)))
df['has_upper_word'] = df['product_title'].apply(lambda x: int(any(word.isupper() for word in x.split())))
brands_list = [
    'iPhone', 'Samsung', 'Sony', 'Philips', 'Bosch',
    'LG', 'Panasonic', 'Apple', 'Dell', 'HP', 'Lenovo', 'Whirlpool'
]
df['has_brand'] = df['product_title'].apply(lambda x: int(any(brand.lower() in x.lower() for brand in brands_list)))


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


X = df[["product_title", "product_name_len", "number_in_name", "has_brand"]]
y = df["category_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[("title", TfidfVectorizer(), "product_title"),
                  ("lenght", MinMaxScaler(), ["product_name_len"]),
                  ("number", MinMaxScaler(), ["number_in_name"]),
                  ("brand", MinMaxScaler(), ["has_brand"])]
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": LinearSVC()
}

for name,model in models.items():
  print(f"\n 🔹 {model}")
  pipeline = Pipeline([
      ("preprocessing", preprocessor ),
      ("classifier", model)])
  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_test)
  print(classification_report(y_test, y_pred))

  import joblib 
X = df[["product_title", "product_name_len", "number_in_name", "has_brand"]]
y = df["category_label"]

preprocessor = ColumnTransformer(
    transformers=[("title", TfidfVectorizer(), "product_title"),
                  ("lenght", MinMaxScaler(), ["product_name_len"]),
                  ("number", MinMaxScaler(), ["number_in_name"]),
                  ("brand", MinMaxScaler(), ["has_brand"])]
)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])
pipeline.fit(X,y)
joblib.dump(pipeline, "category_prediction_model.pkl")
 
print("Model je treniran i sačuvan kao: 'category_prediction_model.pkl'")
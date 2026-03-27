import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# load dataset
df = pd.read_excel("house_price_data.xlsx")
df = df.dropna()

# features and target
X = df[["area", "bedRoom", "bathroom","balcony"]]
y = df["price"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
st.title("🏠 House Price Predictor")

area = st.number_input("Enter Area (sq ft)")
bed = st.number_input("Enter Bedrooms")
bath = st.number_input("Enter Bathrooms")
bal = st.number_input("Enter Balcony")

if st.button("Predict Price"):
    input_data = pd.DataFrame([[area, bed, bath, bal]],
                              columns=["area", "bedRoom", "bathroom", "balcony"])
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Price: {round(prediction[0], 2)} Lakhs")
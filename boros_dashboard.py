import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

st.title("🔮 Boros Energy Winrate Predictor")
st.write("Upload your match data and predict future winrates.")

uploaded_file = st.file_uploader("Upload your matches CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

    # Encode categorical features (including deck_version and meta_shift if present)
    cat_features = ['opponent_archetype', 'event_type']
    if 'deck_version' in df.columns:
        cat_features.append('deck_version')
    if 'meta_shift' in df.columns:
        cat_features.append('meta_shift')

    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[cat_features])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Combine features
    X = pd.concat([df[['days_since_start']], encoded_df], axis=1)
    y = df['winrate']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    st.success("Model trained! You can now make predictions below.")

    # Prediction form
    days = st.slider("Days since first match", 0, int(df['days_since_start'].max()) + 30, 30)
    sel_dict = {}
    for col in cat_features:
        sel_dict[col] = st.selectbox(col.replace("_", " ").title(), df[col].unique())

    # Encode input
    input_df = pd.DataFrame({k: [v] for k, v in sel_dict.items()})
    input_encoded = encoder.transform(input_df)
    input_features = pd.concat([
        pd.DataFrame({'days_since_start': [days]}),
        pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out())
    ], axis=1)

    if st.button("Predict Winrate"):
        prediction = model.predict(input_features)[0]
        st.metric(label="Predicted Winrate", value=f"{prediction:.2f}%")
else:
    st.info("Please upload your CSV data to continue.")
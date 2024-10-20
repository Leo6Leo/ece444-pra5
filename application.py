import pickle

from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer

application = app = Flask(__name__)

# Load the saved model and vectorizer
with open("basic_classifier.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("count_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    # Transform the input text using the loaded vectorizer
    transformed_text = vectorizer.transform([text])

    # Predict using the loaded model
    prediction = loaded_model.predict(transformed_text)

    # Output will be 'FAKE' if fake, 'REAL' if real
    # if prediction[0] == 0:  # Assuming 0 is fake, 1 is real in the model
    #     result = "FAKE"
    # else:
    #     result = "REAL"

    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(debug=True)

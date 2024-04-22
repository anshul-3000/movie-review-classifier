from flask import Flask, render_template, request
import pickle

vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
model = pickle.load(open("model/model.pkl", "rb"))

# Creating instance
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
        review_text = request.form.get("content")
        vectorized = vectorizer.transform([review_text])
        prediction = model.predict(vectorized)
        prediction = 1 if prediction == 1 else 0
        return render_template("index.html", prediction=prediction, review_text=review_text)

if __name__ == "__main__":
    app.run(debug=True)

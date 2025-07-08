from flask import Flask, request, render_template
from recommender import get_recommendations, df1

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    input_text = ""
    if request.method == "POST":
        input_text = request.form["sentence"]
        recommendations = get_recommendations(input_text)
    return render_template("index.html", input_text=input_text, recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

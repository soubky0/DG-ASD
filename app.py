from flask import Flask, render_template, request, jsonify

from test import model_test
import os

app = Flask(__name__)


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file.filename.endswith(".wav"):
            try:
                dest = (
                    os.path.dirname(os.path.realpath(__file__))
                    + "/dev_data/raw/gearbox/uploads/"
                )
                uploaded_file.save(dest + uploaded_file.filename)
                return jsonify({"message": "File uploaded successfully"})
            except Exception as e:
                return str(e)
        else:
            return (
                jsonify({"message": "Invalid file format. Please upload a .wav file"}),
                400,
            )


@app.route("/test", methods=["POST"])
def test():
    if request.method == "POST":
        try:
            model_test()
            return jsonify({"message": "Model tested successfully"})
        except Exception as e:
            return str(e), 400


if __name__ == "__main__":

    app.run(debug=True)

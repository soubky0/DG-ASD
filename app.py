from flask import Flask, render_template, request, jsonify
from model import demo
from enum import Enum
import os

app = Flask(__name__)

class Model(Enum):
    BASELINE = ""
    TIME_MASK = "Augmentations.TIME_MASK_0"
    FREQ_MASK = "Augmentations.FREQUENCY_MASK_0"
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
                uploaded_file.save(dest + "test.wav")
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
        if request.form:
            form_data = request.form.to_dict()
            options = form_data["options"]
            match options :
                case "baseline":
                    model = Model.BASELINE
                case "timemask":
                    model = Model.TIME_MASK
                case "freqmask":
                    model = Model.FREQ_MASK
            result = demo(model.value)
            return jsonify({"message": result})

if __name__ == "__main__":

    app.run(debug=True)

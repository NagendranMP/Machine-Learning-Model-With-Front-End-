#! C:\Users\Nagendran\Desktop\kidney_Disease_front_end\myenv\Scripts\jupyter.exe
from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np

app = Flask(__name__)
load_model = pickle.load(open("after.pkl", "rb"))
sc = joblib.load("scalar.pkl")

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("first.html")

@app.route("/conv", methods=["POST", "GET"])
def conv():
    if request.method == "POST":
        bsl = request.form.get("bsl")
        bu = request.form.get("bu")
        scr = request.form.get("sc")
        pcv = request.form.get("pcv")
        wbc = request.form.get("wbc")
        trans = sc.transform([[bsl, bu, scr, pcv, wbc]])
        raw = load_model.predict(trans)
        print(type(raw[0]))
        if raw[0] == 0:
            pred = "Sorry To Say This You will Have Chronic Kidney Disease"
        elif raw[0] == 2:
            pred = "Congrats You Won't Have Chronic Kidney Disease"
        elif raw[0] == 1:
            pred = "You Are A Rare Case So You May HAVE or NOT"
        return render_template("result.html", prediction=pred, bs1=bsl, bu1=bu, scr1=scr, pcv1=pcv, wbc1=wbc)
    return render_template("result.html", prediction=None, bs1=None, bu1=None, scr1=None, pcv1=None, wbc1=None)

if __name__ == "__main__":
    app.run()

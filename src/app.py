from x_ray_api import COVID19_Classification
from flask import Flask, jsonify, request, render_template
import os
app = Flask(__name__)
x_ray = COVID19_Classification()


@app.route("/")
def home():
    return render_template('home.html')


# @app.route("/get_data")
# def get_data():
#     x_ray.get_data()
#     return render_template('get_data.html', text='Success', color='success')


# @app.route("/preprocessing")
# def preprocess():
#     x_ray.preprocess()
#     return render_template('preprocess.html', text='Success', color='success')


# @app.route("/training")
# def train():
#     x_ray.train()
#     return render_template('result.html', text='Success', color='success')


@app.route("/predict")
def predict():
    arr = x_ray.predict()
    return render_template('predict.html', p1=arr[0], p2=arr[1], p3=arr[2], p4=arr[3], p5=arr[4], color='success')


@app.route("/predict_html")
def predict_html():
    return render_template('predict_path.html')


@app.route("/predict_path")
def predict_path():
    path = request.args['text']
    res = x_ray.predict_path(path)
    return render_template('predict_show.html', text=res, color='success')


app.run()

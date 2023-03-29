from NeuralNetwork import NeuralNetwork

import base64
from io import BytesIO

from flask import Flask, render_template
from matplotlib.figure import Figure

app = Flask(__name__)

nn = NeuralNetwork(10)

@app.route("/")
def main():
    return ''' 
        <h1>MNIST Dataset</h1>
        <a href="/dataset">Load Dataset</a>
        '''

@app.route("/dataset")
def initialiseDataset():
    nn.create()
    return '''
        <h1>Options List</h1>
        <a href="/predict-images">Predict Images</a>
        <a href="/predict-random-image">Predict Random Image</a>
        <a href="/confusion-matrix">Confusion Matrix</a>
        <a href="/show-errors">Show Errors</a>
    '''

@app.route("/predict-images")
def predictImages():
    plot = nn.predictImages()
    return render_template("plot.html", url="/static/images/predict-images.png")

@app.route("/predict-random-image")
def predictRandomImage():
    nn.prepareTrainingData()
    nn.train()
    nn.evaluate()
    plot = nn.predictSingle()
    return render_template("plot.html", url="/static/images/predict-random-image.png")

@app.route("/confusion-matrix")
def confusionMatrix():
    nn.prepareTrainingData()
    nn.train()
    nn.evaluate()
    plot = nn.confusionMatrix()
    return render_template("plot.html", url="/static/images/confusion-matrix.png")

@app.route("/show-errors")
def showErrors():
    nn.prepareTrainingData()
    nn.train()
    nn.evaluate()
    plot = nn.showErrors()
    return render_template("plot.html", url="/static/images/show-errors.png")

'''
def hello():
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"
'''

app.run("127.0.0.1", 5000, True)
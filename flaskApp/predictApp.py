# Serve model as a flask application
import pickle
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, render_template
from torchvision import datasets, transforms, models


app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        image = image_loader(file)

        ps = torch.exp(model(image))
        _, predTest = torch.max(ps, 1)
        print(ps.float().round())
        result = "Your Image is probably a "
        if predTest == 0:
            result += " Cat"
        else:
            result += " Dog"
        return result

    return render_template('home.html')


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
load_model()

# defining image loader


def image_loader(image_name):  # load the image from image loader
    imsize = 256
    loader = transforms.Compose(
        [transforms.Scale(imsize), transforms.ToTensor()])
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image.cuda()

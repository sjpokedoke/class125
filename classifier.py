import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

xtrainscaled = xtrain/255.0
ytrainscaled = ytrain/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xtrainscaled, ytrain)

def getPrediction(image):
    im_pil = Image.open(image)
    imbw = im_pil.convert("L")
    imbwresize = imbw.resize((28, 28), Image.ANTIALIAS)
    
    pixelfilter = 20
    minpixel = np.percentile(imbwresize, pixelfilter)

    imbwresizeinvertedscaled = np.clip(imbwresize - minpixel, 0, 255)

    maxpixel = np.max(imbwresize)

    imbwresizeinvertedscaled = np.asarray(imbwresizeinvertedscaled)/maxpixel
    
    testsample = np.array(imbwresizeinvertedscaled).reshape(1, 784)
    testpred = clf.predict(testsample)

    return testpred[0]
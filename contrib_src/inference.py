#!/usr/bin/env python -W ignore::DeprecationWarning
import json
from keras.models import load_model
from processing import ImageProcessor
from modelhublib.model import ModelBase

class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the model with keras
        self._model = load_model('model/model.h5')
        self._model._make_predict_function()
    

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # Run inference with keras
        results = self._model.predict(inputAsNpArr)
        # postprocess results into output
        output = self._imageProcessor.computeOutput(results)
        return output
        


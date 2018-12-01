import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import pandas as pd
from sqlalchemy import create_engine
from PIL import Image

import time

def load_checkpoint(filename):
    '''
    inputs the .pth file and outputs model
    '''
    if torch.cuda.is_available():
        checkpoints = torch.load(filename)
    else:
        checkpoints = torch.load(filename, map_location = 'cpu')
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 4096)),
    ('relu1', nn.ReLU()),
    ('fc4', nn.Linear(4096, 2048)),
    ('relu4', nn.ReLU()),
    ('fc5', nn.Linear(2048, 1024)),
    ('relu5', nn.ReLU()),

    ('fc6', nn.Linear(1024, 57)),
    ('output', nn.LogSoftmax(dim = 1))
    ]))
    model.classifier = classifier
    model.class_to_idx = checkpoints['class_to_idx']
    model.load_state_dict(checkpoints['state_dict'])
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    size = (256,256)
    image = image.resize(size)
    width, height = image.size
    left = (width - 224) / 2
    right = (width + 224) / 2
    top = (height - 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
    try:
        np_image = np.array(image) / 255

        mean_normal = np.array([0.485, 0.456, 0.406])
        std_normal = np.array([0.229, 0.224, 0.225])

        np_image = (np_image - mean_normal) / std_normal
        np_image = np_image.transpose(2, 0, 1)
    except:
        image = image.convert("RGB")
        np_image = np.array(image) / 255

        mean_normal = np.array([0.485, 0.456, 0.406])
        std_normal = np.array([0.229, 0.224, 0.225])

        np_image = (np_image - mean_normal) / std_normal
        np_image = np_image.transpose(2, 0, 1)
    return np_image

def predict(image_path, model, topk=10):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(image_path)

    img = torch.from_numpy(image).type(torch.FloatTensor)
    img.unsqueeze_(0)
    model.to('cpu')
    output = model.forward(img)
    probs = torch.exp(output)
    top_5_probs, top_5_labels = probs.topk(5)
    return top_5_probs, top_5_labels
    # TODO: Implement the code to predict the class from an image file

def create_label(classes, model):
    '''
    this function convert labels from tensor to list
    input classes and model
    returns label in the form of list
    '''

    #convert tensor to array, and array to list
    labels_list = classes.numpy()
    labels_list = labels_list.tolist()
    labels = []
    for i in labels_list:
        for j in i:
            labels.append(j)

    #creating dictionary from class_to_idx
    model_dict = model.class_to_idx
    #switch the keys and values from the dictionary
    model_dict2 = {y:x for x,y in model_dict.items()}

    #convert the list element in labels into the values in model dict
    convert_list = []
    for i in labels:
        convert_list.append(model_dict2[i])
    return convert_list

def create_probs(probs):
    '''
    this function convert probs from tensor to list
    converting probs from tensor to ndarray and ndarray to list
    '''
    probs_array = probs.detach().numpy()
    probs_list1 = probs_array.tolist()
    probs_list = []
    for i in probs_list1:
        for j in i:
            probs_list.append(j)
    return probs_list

#img = './img/test.jpg'
import json
def solve(img):
    '''
    This function is used to predict Image
    '''
    start_time = time.clock()
    #load model
    model_new = load_checkpoint('checkpoints.pth')

    #load image
    image_path = img
    image = process_image(image_path)

    #predict probability
    probs, classes = predict(image_path, model_new)

    #create labels and probabilities as lists
    labels_use = create_label(classes, model_new)
    probs_use = create_probs(probs)

    #sort label based on probabilities
    #Z = [x for _,x in sorted(zip(probs_use,labels_use), reverse = True)]
    dict_json = dict()
    #Z = dict(zip(labels_use,probs_use))
    #dict_json['data'] = Z
    #dict_json['time'] = time.clock() - start_time
    #dict_json = json.dumps(dict_json)


    test_list = []
    for i in range(len(labels_use)):
        dict_test = dict()
        dict_test['name'] = labels_use[i]
        dict_test['pred_accuracy'] = probs_use[i]
        test_list.append(dict_test)
    test_list = sorted(test_list, key = lambda x: list(x.values()), reverse = True)
    dict_json['data'] = test_list
    dict_json['time'] = time.clock() - start_time
    dict_json = json.dumps(dict_json)
    return dict_json

#Z = solve(img)
#print(Z)
#create a dataframe
#data = [(image_path, Z[0], Z[1], Z[2], Z[3], Z[4])]
#df = pd.DataFrame(data, columns = ['path', 'p1', 'p2', 'p3', 'p4', 'p5'])


#engine = create_engine('mysql+pymysql://root:agus123!@#@localhost:3306/admincerdas')
#df.to_sql(name = 'predict_image',
#         con = engine,
#         index = False,
#         if_exists = 'append')

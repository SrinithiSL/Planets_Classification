import os
import uuid
import flask
import urllib
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable 
from flask import Flask , render_template  , request , send_file


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = torch.jit.load(os.path.join(BASE_DIR ,'model_scripted.pt'))
model.eval()


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

classes,class_to_idx=find_classes('Planets and Moons')
print(classes,class_to_idx)

from numpy import exp
 
# calculate the softmax of a vector
def softmax(vector):
 e = exp(vector)
 return e / e.sum()



def predict(img_path , model):
    image= Image.open(img_path)
    transformer= transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    image_tensor= transformer(image).float()
    image_tensor= image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
     image_tensor.cuda()
    
    input=Variable(image_tensor)
    input = input.to(device)

    output= model(input)
    index= output.data.argmax()
    pred= classes[index]

    result = output.detach().numpy()
    print(result)
    dict_result = {}
    for i in range(11):
        dict_result[result[0][i]] = classes[i]
    
    res = result[0]
    #res.sort()
    #res = res[::1]
    print(res)
    
    indices=np.argsort(res)[::-1][:3]
    prob = res[indices]
    prob_result = []
    class_result=[]
    for i in range(3):
        prob_result.append((prob[i]*100).round())
        class_result.append(dict_result[prob[i]])
    

    return class_result , prob_result

    
 



@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(port=8080,debug = True)



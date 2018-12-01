# from keras.models import load_model
from flask import Flask
from flask import request
# import tensorflow as tf
import predict_image_method
# import urllib.request
import requests
import time
# global graph, model
# print(Flask.__version__)

#IMG_FOLDER = "E:\\Admincerdas\\foto_flask\\"
IMG_FOLDER = "/var/sentora/hostdata/zadmin/public_html/python/image_prediction/"
# MODEL_FILENAME = "captcha_model.hdf5"
# model = load_model(MODEL_FILENAME)
# graph = tf.get_default_graph()

app = Flask(__name__)

@app.route('/', methods=["GET"])
@app.route('/index', methods=["GET"])
def index():
	# urllib.request.urlretrieve("https://ib.bri.co.id/ib-bri/login/captcha", "test/download.jpg")
	img = request.args.get("img")
	print(img)
	name = str(int(time.time())) + ".png"
	# f = open("test/"+name,'wb')
	with open(IMG_FOLDER+"test/"+name, 'wb') as f:
		f.write(requests.get(img).content)
    	# f.write(requests.get('https://ib.bri.co.id/ib-bri/login/captcha').content)
    	# f.close()

	# print(IMG_FOLDER+img)
	# call method from other script
	# result = solve_captchas_with_model.solve(IMG_FOLDER+img)
	result2 = predict_image_method.solve(IMG_FOLDER+"test/"+name)
	#print(result2)
	# return result+" "+result2
	return result2

if __name__ == "__main__":
	# app.run()
	app.run(host='0.0.0.0', port = 5001)
	# app.run(debug=False)

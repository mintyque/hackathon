# app.py

from flask import Flask, request, render_template, Response
import json

#car model classification
from classifier import predict

#car plate detection

#car plate recognition
from recognizer import recognize

app = Flask(__name__)



@app.route('/upload', methods=['GET', 'POST'])
def service():
	if request.method == 'POST':
		file = request.files['file']
		
		# Car model classification
		#format: LADA_PRIORA_B
		brand, model, veh_type = predict('image_test.jpg')
		# Car plate detection
		#plate_image = detect('image_path')

		#Car plate recognition
		car_plate = recognize(plate_image) 

		response = {"brand":brand,"model":model,"probability":"72.5","veh_type":veh_type,"coord":"[(398,292),(573,360)]","id":"0001","plate":"x000xxx111"}
		response = json.dumps(response)

		return Response(response=response, status=200, mimetype="application/json")	
	return render_template("service.html")


# We only need this for local development.
if __name__ == '__main__':
	app.run()

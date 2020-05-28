# app.py

from flask import Flask, request, render_template, Response
import json
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def service():
	if request.method == 'POST':
		file = request.files['file']
		#TODO: work with neural networks
		response = {"brand":"RENAULT","model":"DUSTER","probability":"72.5","veh_type":"B","coord":"[(398,292),(573,360)]","id":"0001","plate":"x000xxx111"}
		response = json.dumps(response)
		return Response(response=response, status=200, mimetype="application/json")	
	return render_template("service.html")

# We only need this for local development.
if __name__ == '__main__':
	app.run()

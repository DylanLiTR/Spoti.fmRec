from flask import Flask, render_template, request
from spotifyrec import start
import pandas as pd
import threading

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		recs = start(request.form.get("username"))
		return recs.to_html(index=False)
		
	return render_template("index.html")

class MyWorker():
	def __init__(self, username):
		self.username = username

		thread = threading.Thread(target=self.run, args=())
		thread.daemon = True
		thread.start()

	def run(self):
		start(self.username)
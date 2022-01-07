from flask import Flask, render_template, request
from spotifyrec import start
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		recs = start(request.form.get("username"))
		return recs.to_html(index=False)
		
	return render_template("index.html")
from flask import Flask

app = Flask(__name__)

from starbucks_analysis_app import routes

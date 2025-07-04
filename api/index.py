import awsgi
from webapp import app as flask_app


def handler(event, context):
    return awsgi.response(flask_app, event, context) 
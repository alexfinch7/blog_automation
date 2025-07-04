from vercel_wsgi import handle_request
from webapp import app as flask_app


def handler(event, context):
    # Vercel invokes this function for each request.
    return handle_request(flask_app, event, context) 
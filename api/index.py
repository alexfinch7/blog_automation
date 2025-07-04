import awsgi
from webapp import app

def handler(event, context):
    """
    This is the entrypoint Vercel will call.
    awsgi.response will translate the Lambda event+context
    into a WSGI request for Flask, and back into a Lambda response.
    """
    return awsgi.response(app, event, context)
# api/index.py

# Restrict exports so Vercel only sees `handler`
__all__ = ["handler"]

def handler(event, context):
    """
    Entrypoint for Vercel’s Python runtime.
    Translates Lambda-style events into WSGI requests for Flask.
    """
    import awsgi
    from webapp import app

    # awsgi.response wraps your Flask app in a Lambda-compatible response
    return awsgi.response(app, event, context)

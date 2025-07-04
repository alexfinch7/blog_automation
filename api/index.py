# api/index.py

# Tell Python/Vercel “__all__ only contains handler”
__all__ = ["handler"]

def handler(event, context):
    """
    Vercel will invoke this.  We import AWGI
    and your Flask `app` here so that no other
    symbols escape into module scope.
    """
    import awsgi
    from webapp import app

    # Turn the Lambda-style event/context into a WSGI request,
    # dispatch it to your Flask app, and return the response.
    return awsgi.response(app, event, context)

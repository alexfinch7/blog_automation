__all__ = ["handler"]

def handler(event, context):
    import awsgi
    from webapp import app

    return awsgi.response(app, event, context)
from webapp import app

# For Vercel deployment, we need to expose the Flask app instance
# This is the standard pattern for Vercel Python deployments
if __name__ == "__main__":
    app.run()
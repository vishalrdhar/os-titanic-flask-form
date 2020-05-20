from flask import Flask
application = Flask(__name__)

@application.route("/")
def hello():
    return "Hello Ray! I am talking to you from an app deployed via okd and github"

@application.route("/test")
def hello_test():
    return "Hello Ray! You added test to the URL!"

if __name__ == "__main__":
    application.run()

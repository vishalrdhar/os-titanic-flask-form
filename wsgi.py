#! python3
################################################################################
#
# Author: Award Solutions, Inc.
# Course: ANI_408 Analytics Workshop
# Exercise: ANALYZE_TITANIC_Logistic_Regression_APP.py
#
# ©2018 Award Solutions, Inc. All Rights Reserved. 
# Disclaimer: Award writes scripts/programs and creates visualizations in class
# and out of class (“Tools”) that Award shares “AS IS” with students in Award’s
# training classes, under a single-user, limited, non-exclusive,
# non-transferable license for instructional purposes only. Students shall
# provide their own computers, secure the necessary software and software
# licenses, and satisfy the system and software requirements to run the Tools.
# Award does not warrant any particular outcome from use of the Tools, any
# particular application, or compatibility of the Tools with any system,
# software or equipment. Award is not responsible for any damages caused by the
# use of the Tools. Award retains for itself all right, title and interest in
# and to the Tools, including all copyrights and other intellectual property
# rights. Do not share, sell, transfer, copy, publish, decompile, reverse
# engineer, or create derivative works from the Tools. 
#
################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import os
import scipy.stats as stats

from sklearn.linear_model import LogisticRegression
# from pylab import rcParams
from sklearn.model_selection import train_test_split

import warnings

from sklearn import metrics

# pip install flask
from flask import Flask, render_template, request

warnings.simplefilter("ignore")

global MODEL_BUILT
MODEL_BUILT = False
global MODEL
MODEL = None
global PRED_COUNT
PRED_COUNT = 0

###############################################################################
#
# Function Name: my_main()
# This is the main function of the program
#
###############################################################################
def build_regression_model(filename):
	titanic = pd.read_csv(filename)

	titanic.drop(['PassengerId'], axis=1, inplace = True)
	titanic.drop(['Name'], axis=1, inplace = True)
	titanic.drop(['Ticket'], axis=1, inplace = True)
	titanic.drop(['Cabin'], axis=1, inplace = True)
	titanic['Sex'].replace(['male','female'],[0,1], inplace = True)
	titanic['Embarked'].replace(['C','Q','S'],[0,1,2], inplace = True)

	titanic = titanic.dropna()

	x = titanic.drop(labels = ['Survived'],axis = 1)
	y = titanic['Survived'].values

	#Import the model we want to incorporate for your dataset
	#Fit the data into your model
	logreg = LogisticRegression().fit(x,y)

	prediction = logreg.predict(x)

	# Use score method to get accuracy of model
	# score = logreg.score(x, y)
	# print("Accuracy Score is: %4.2f" %(score))

	cm = metrics.confusion_matrix(y, prediction)
	#corr.style.background_gradient(cmap='coolwarm').set_precision(2)

	TP = cm[0][0]
	FP = cm[0][1]
	FN = cm[1][0]
	TN = cm[1][1]

	return logreg
# END OF FUNCTION


def my_main(input_list):
	
	global MODEL_BUILT
	global MODEL
	global PRED_COUNT
	filename = 'titanic.csv'

	
	if not MODEL_BUILT:
		print("Model NOT Ready ... building now")
		MODEL = build_regression_model(filename)
		MODEL_BUILT = True
	else:
		print("Model Ready ...")
	# end of if
	
	x1 = [input_list]
	pr = MODEL.predict(x1)
	PRED_COUNT += 1
	return pr, filename, PRED_COUNT
# END OF FUNCTION


###############################################################################
#
# This is REALLY the main body of my program
#
###############################################################################
if __name__ == "__main__":
	print("++-->> Python special variable ... __name__::", __name__)
	print("++-->> ANALYZE_TITANIC_Logistic_Regression_APP.py program is being run STANDALONE!!\n\n")
	
	app.run(host='0.0.0.0', debug=True)
else:
	print("++-->> Python special variable ... __name__::", __name__)
	print("++-->> ANALYZE_TITANIC_Logistic_Regression_APP.py program is being called by SOMEONE!!\n\n")
	
	# instantiate the Flask object and then run it
	app = Flask(__name__)
	
	@app.route('/',methods = ['POST', 'GET'])
	def index():
		return render_template('input_template.html')
	# end of function
	
	
	@app.route('/result',methods = ['POST', 'GET'])
	def result():
		if request.method == 'POST':
			result = request.form
			name = result['name']
			Pclass = int(result['Pclass'])
			Sex = int(result['Sex'])
			Age = float(result['Age'])
			SibSp = int(result['SibSp'])
			Parch = int(result['Parch'])
			Fare = float(result['Fare'])
			Embarked = int(result['Embarked'])
			input_list = [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
			pred, f_name, total_count = my_main(input_list)
			
			if pred == 1:
				prediction = "TRUE"
			else:
				prediction = "FALSE"
		
			return render_template('template.html',
									my_string=f_name,
										name=name,
										input=input_list,
											prediction=prediction,
												total_count=total_count)
	# end of function
# end of if else

###############################################################################
# END OF FILE

#
# code copied from oroginal wsgi.py example file
#
# from flask import Flask
# application = Flask(__name__)

# @application.route("/")
# def hello():
    # return "Hello Ray! I am talking to you from an app deployed via okd and github"

# @application.route("/test")
# def hello_test():
    # return "Hello Ray! You added test to the URL!"

# if __name__ == "__main__":
    # application.run()

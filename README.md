Problem and solution:


Football clubs make transfer decisions involving millions of euros, and accurately estimating a player's market value is critical for scouting, negotiations, and career mode simulations.
However, a player’s market value depends on many factors — physical attributes, skills, age, potential, wages, and position — making manual estimation inconsistent and subjective.

This project aims to build a machine learning model that predicts the market value of a football player using FIFA player data from 2021.

The goal is to:

Create a clean dataset that includes key football performance and physical attributes
Build an ML model capable of estimating a player’s true market price
Provide a reusable prediction script for valuing any new player
Compare ML approaches and select the most accurate one

The final solution uses XGBoost, which achieved the best performance (R² ≈ 0.99), making it suitable for real-world player value estimation.

"Some features, such as wage, overall rating, and potential, have extremely strong correlation with market value. As a result, models like XGBoost can achieve very high R². This is realistic given how football player values are decided in practice."

How to use: 
step 1: Download all stuff in requirenment.txt most of them you can easily be downloaded by typed in cmd:  pip install "all things from requirenments" 

step 2: download data Career_Mode_ player datasets_FIFA_15-21.csv

step 3: open jupyter notebook

step 4: copy file Train and paste in first cell of your notebook 

step 5: change directory from where you will open data

step 6: run it

step 7: copy file predic.py and paste it into second cell of your juoyter notebook

step 8: change directory from where you will open data

step 9: run it and it should show the result you can also change feauters for this model (its explained in file)

!!!dont forget to change "data.csv" to your real file name and directory!!!

I want to add that my Eda is file and it can be easily opened in visual studio code

Some additional information about this project:

In this project have been used three models Linear regression, RandomForestRegression and XGBOOSTREGRESSION
the most acurate is xgboost but randomforst is very close to it too. My model has very high accurate of prediction because some feauters are straight affect on players value 






from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import mysql.connector
import json
app = Flask(__name__)
CORS(app)

# Connect to the MySQL database using XAMPP
# mydb = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   password="",
#   database="corruption"
# )

@app.route('/api/get_data', methods=['GET'])
def get_data():
    data = [
        {'name': 'John', 'age': 30, 'city': 'New York'},
        {'name': 'Jane', 'age': 25, 'city': 'San Francisco'},
        {'name': 'Bob', 'age': 40, 'city': 'Chicago'},
    ]
    return jsonify(data)

# @app.route('/api/provinsi', methods=['GET'])
# def get_data_provinsi():
#     mycursor = mydb.cursor()
#     mycursor.execute("SELECT * FROM provinsi")
#     data = mycursor.fetchall()

#     result = []
#     for row in data:
#         d = {'provinsiId': row[0], 'provinsiName': row[1]}  
#         result.append(d)


#     # Return data in JSON format
#     return jsonify(result)

# @app.route('/api/kabupaten', methods=['GET'])
# def get_data_kabupaten():
#     mycursor = mydb.cursor()
#     mycursor.execute("SELECT * FROM kabupaten")
#     data = mycursor.fetchall()

#     resultkabupaten = []
#     for row in data:
#         dkabupaten = {'kabupatenId': row[1], 'kabupatenName': row[2]}  
#         resultkabupaten.append(dkabupaten)

#     # Return data in JSON format
#     return jsonify(resultkabupaten)
import pandas as pd
import pickle
import numpy as np
@app.route('/api/submit_data', methods=['POST'])
def submit_data():
    # Load the trained KNN model from a file
    with open('knn_Final_model_json.pkl', 'rb') as file:
        knn = pickle.load(file)



    form_data = request.json

    data = {}
    for key, value in form_data.items():
        if key.startswith('pertanyaan'):
            data[key] = [value]
    # pertanyaan1 = form_data['pertanyaan1']
    # pertanyaan2 = form_data['pertanyaan2']
    # pertanyaan3 = form_data['pertanyaan3']
    # pertanyaan4 = form_data['pertanyaan4']
    # pertanyaan5 = form_data['pertanyaan5']

    # # Create a DataFrame from the form data
    # data = {
    #     'pertanyaan1': [pertanyaan1],
    #     'pertanyaan2': [pertanyaan2],
    #     'pertanyaan3': [pertanyaan3],
    #     'pertanyaan4': [pertanyaan4],
    #     'pertanyaan5': [pertanyaan5]
    # }
    df = pd.DataFrame(data)

    # Make a prediction using the KNN model
    y_pred = knn.predict(df)

    # Convert the prediction value to a JSON string
    json_data = json.dumps({'prediction': y_pred.tolist()})

    # # Return the processed data to the client
    return json_data

if __name__ == '__main__':
    app.run(debug=True)

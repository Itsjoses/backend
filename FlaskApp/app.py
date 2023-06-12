from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
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

import random
@app.route('/api/Generate_data', methods=['POST'])
def Generate_data():

    form_data = request.json

    my_file = pd.read_csv('merge-table.csv',header=None)
    wf = open("dataset.txt","w")
    dimension = form_data.get('Dimension')
    arr = [0] * dimension
    file = open("jsondata.json", "w")
    file.write("["+"\n")



    for i in range(1,515):
        for j in range(1,50):
            file.write("{")
            total = 0
            for k in range(0,dimension):
                arr[k] = random.randint(1,10)
                total += arr[k]
            result = total/(dimension * 10)*100
            
            file.write('"provinsi" : "')

            file.write(str(my_file[1].iloc[i]))
                    
            file.write('",')

            file.write('"kabupaten" : "')

            file.write(str(my_file[2].iloc[i]))
                    
            file.write('",')

            for k in range(0,dimension):
                file.write('"pertanyaan'+str(k+1)+'" : ')

                file.write(str(arr[k]))
                        
                file.write(',')

            if (result >= 0 and result < 20):
                file.write('"Result" : 1')
            elif (result >= 20 and result < 40):
                file.write('"Result" : 2')
            elif (result >= 40 and result < 60):
                file.write('"Result" : 3')
            elif (result >= 60 and result < 80):
                file.write('"Result" : 4')
            elif (result >= 80 and result <= 100):
                file.write('"Result" : 5')

            if i == 514 and j == 49:
                file.write("}"+"\n")
            else:
                file.write("},"+"\n")
        print(i)  
    file.write("]")
    json_data = json.dumps({'Status': "Done"})
    return json_data

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import timeit

@app.route('/api/Generate_Model', methods=['POST'])
def Generate_Model():
    with open('jsondata.json', 'r') as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    X = df.drop(['provinsi', 'kabupaten', 'Result'], axis=1)
    y = df['Result'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Find the optimal number of neighbors
    neighbors = range(1, 41)
    train_acc = []
    test_acc = []

    # Create KNN classifier with optimal k
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: {:.2f}%".format(accuracy*100))

    with open('knn_Final_model_json.pkl', 'wb') as file:
        pickle.dump(knn, file)
    
    json_data = json.dumps({'Status': "Done"})
    return json_data

if __name__ == '__main__':
    app.run(debug=True)

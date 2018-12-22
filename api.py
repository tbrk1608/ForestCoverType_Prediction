from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def index():
    """
    sample input:
        
    {
       "Elevation":"2680","Aspect":"354","Slope":"14".......
    }
    
    Please enter all the 55 columns data like in the test.csv file
    Incase of missing data enter "0"
    """
    input_data = request.json
    data = []
    for key,value in input_data.items():
        if key in cols:
            data.append(value)
    data = np.array(data).reshape(1,52)
    pred = model.predict(data)
    return jsonify(str(pred))
       

if __name__ == '__main__':
	model = pickle.load(open("model.pkl","rb"))
	cols = pickle.load(open("model_columns.pkl","rb"))
	app.run(debug=True)
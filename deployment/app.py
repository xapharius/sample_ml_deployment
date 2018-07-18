import sys
from flask import Flask
from flask import request
from flask import jsonify
from flask import Response

import joblib
import pandas as pd
sys.path.append("src")


class Predictor(object):
    
    def __init__(self, pkl_path):
        self.preprocessor = joblib.load(pkl_path + '/preprocessor.pkl')
        self.model = joblib.load(pkl_path + '/model.pkl') 
    
    def predict(self, df):
        df = self.preprocessor.transform(df)
        preds = self.model.predict_proba(df)[:,1]
        return preds
    

def parse_request(data):
    if isinstance(data, dict):
        data = [data]
    
    try:
        df = pd.DataFrame(data)
    except:
        raise ValueError("Invalid request json")
        
    valid_columns = ["pclass", "name", "sex", "age", "sibsp", "parch", 
                     "ticket", "fare", "cabin", "embarked"]
    if not (set(df.columns) <= set(valid_columns)):
        raise ValueError("Invalid features")
    return df


app = Flask(__name__)
predictor = Predictor("./pkl")


@app.route('/', methods=['POST'])
def predict():
    raw = request.get_json()
    
    try:
        df = parse_request(raw)
    except ValueError as e:
        return Response("Malformed request", status=400)
    preds = predictor.predict(df)
    return jsonify(list(preds))


if __name__ == "__main__":
    app.run()
    

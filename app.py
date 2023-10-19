from flask import Flask,request,jsonify
import pandas as pd
import pickle
# #import seaborn as sns
from flask_cors import CORS
import json
app=Flask(__name__)
CORS(app)


model=pickle.load(open('nifty.pkl','rb'))
# data=[[25850.36,22333.33,24685.89,21865.99]]

# df=pd.DataFrame(data,columns=['Open','High','Low','Close'])
# print(df.head())
# x=model.predict(df)
# print(x)
@app.route('/prediction',methods=['POST'])
def prediction():
    data=request.json
    Open=data['open']
    High=data['high']
    Low=data['low']
    
    x=[[Open,High,Low]]
    df=pd.DataFrame(x,columns=['Open','High','Low'])
    output=model.predict(df)
    response={"output":round(output[0],2)}
    return jsonify(response)



if __name__ == "__main__":
    app.run(debug=True,port=5000)
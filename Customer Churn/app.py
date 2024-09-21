from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [
            float(request.form['Tenure']),
            int(request.form['PreferredLoginDevice']),
            int(request.form['CityTier']),
            float(request.form['WarehouseToHome']),
            int(request.form['PreferredPaymentMode']),
            int(request.form['Gender']),
            float(request.form['HourSpendOnApp']),
            int(request.form['NumberOfDeviceRegistered']),
            int(request.form['PreferedOrderCat']),
            int(request.form['SatisfactionScore']),
            int(request.form['MaritalStatus']),
            int(request.form['NumberOfAddress']),
            int(request.form['Complain']),
            float(request.form['OrderAmountHikeFromlastYear']),
            int(request.form['CouponUsed']),
            int(request.form['OrderCount']),
            float(request.form['DaySinceLastOrder']),
            float(request.form['CashbackAmount'])
        ]
        
        # Convert features to a numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Convert numerical prediction to 'Yes' or 'No'
        result = 'Yes' if prediction == 1 else 'No'
        
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'PredictedChurn': result})
        else:
            return render_template('result.html', prediction=result)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'error': str(e), 'status': 500}), 500
        else:
            return render_template('error.html', error=str(e))

# Custom error handler for 405 Method Not Allowed
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed', 'status': 405}), 405

# Custom error handler for general exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"An error occurred: {str(e)}")
    return jsonify({'error': str(e), 'status': 500}), 500

if __name__ == '__main__':
    app.run(debug=True)

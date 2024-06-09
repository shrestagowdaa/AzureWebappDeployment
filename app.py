import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
# Load the trained voting classifier model
model = joblib.load(open('voting_classifier_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    
    # Convert categorical features to numeric
    final_features = [
        1 if int_features[0] == 'Male' else 0,  # gender
        int(int_features[1]),  # senior_citizen
        1 if int_features[2] == 'Yes' else 0,  # partner
        1 if int_features[3] == 'Yes' else 0,  # dependents
        int(int_features[4]),  # tenure
        1 if int_features[5] == 'Yes' else 0,  # phone_service
        2 if int_features[6] == 'Yes' else (1 if int_features[6] == 'No phone service' else 0),  # multiple_lines
        2 if int_features[7] == 'Fiber optic' else (1 if int_features[7] == 'DSL' else 0),  # internet_service
        2 if int_features[8] == 'Yes' else (1 if int_features[8] == 'No internet service' else 0),  # online_security
        2 if int_features[9] == 'Yes' else (1 if int_features[9] == 'No internet service' else 0),  # online_backup
        2 if int_features[10] == 'Yes' else (1 if int_features[10] == 'No internet service' else 0),  # device_protection
        2 if int_features[11] == 'Yes' else (1 if int_features[11] == 'No internet service' else 0),  # tech_support
        2 if int_features[12] == 'Yes' else (1 if int_features[12] == 'No internet service' else 0),  # streaming_tv
        2 if int_features[13] == 'Yes' else (1 if int_features[13] == 'No internet service' else 0),  # streaming_movies
        2 if int_features[14] == 'Two year' else (1 if int_features[14] == 'One year' else 0),  # contract
        1 if int_features[15] == 'Yes' else 0,  # paperless_billing
        3 if int_features[16] == 'Credit card (automatic)' else (2 if int_features[16] == 'Bank transfer (automatic)' else (1 if int_features[16] == 'Mailed check' else 0)),  # payment_method
        float(int_features[17]),  # monthly_charges
        float(int_features[18])  # total_charges
    ]
    
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)
    print(prediction[0])

    return render_template('home.html', prediction_text="Prediction: {}".format(prediction[0]))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

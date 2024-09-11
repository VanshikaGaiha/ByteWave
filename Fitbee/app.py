from flask import Flask, render_template, request,redirect, url_for,jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier




app = Flask(__name__)

# Preload the dataset (you can adjust the file path)
file_path = './Upload/Final_Exercise_Recommendations_Dataset new.csv'
try:
    data = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

# Preprocess the data
data['Exercise Recommendation Plan'] = data['Exercise Recommendation Plan'].astype('category')
X = data[['Weight', 'Height', 'BMI', 'Gender', 'Age', 'BMIcase']]
X = pd.get_dummies(X, drop_first=True)
y = data['Exercise Recommendation Plan'].cat.codes

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the CatBoost Classifier
model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)

# Flask route for prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input JSON from the frontend
    input_data = request.json

    # Parse the input JSON into the necessary features
    weight = input_data.get('Weight')
    height = input_data.get('Height')
    bmi = input_data.get('BMI')
    gender = input_data.get('Gender')  # 'Male' or 'Female'
    age = input_data.get('Age')
    bmi_case = input_data.get('BMIcase')  # Assuming this is categorical like 'Underweight', 'Normal', etc.

    # Create a DataFrame for the new input
    new_data = pd.DataFrame({
        'Weight': [weight],
        'Height': [height],
        'BMI': [bmi],
        'Gender': [gender],
        'Age': [age],
        'BMIcase': [bmi_case]
    })

    # One-hot encode the new input (matching the training data)
    new_data_encoded = pd.get_dummies(new_data, drop_first=True)

    # Ensure the new data has the same columns as the training set (even if some features are missing)
    new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Make a prediction
    prediction = model.predict(new_data_encoded)

    # Convert the prediction back to a readable format if necessary
    exercise_plan = data['Exercise Recommendation Plan'].cat.categories[prediction[0]]

    # Return the result as a JSON response
    return jsonify({
        'Exercise Recommendation Plan': exercise_plan
    })


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
       # Here, you would handle saving data to the database.
        # For now, we'll skip straight to redirecting.
        return redirect(url_for('index'));
    return render_template('Login.html')
   

if __name__ == '__main__':
    app.run(debug=True)

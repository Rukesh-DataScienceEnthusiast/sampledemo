import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from flask import Flask, render_template, request

excel_file_path = r"P:\Datasets\rec_sys.xlsx"
df1 = pd.read_excel(excel_file_path, index_col=0)
print(df1)

# Encoding categorical labels using LabelEncoder
encoder = LabelEncoder()

# Encoding all remaining categorical columns
categorical_columns = df1.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    df1[col] = encoder.fit_transform(df1[col])

# Prepare X and Y
X = df1.drop(['pg course'], axis=1)
Y = df1['pg course']
print(X)
print(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Create KNN model
knn = KNeighborsClassifier()

# Train the model
knn.fit(X_train, Y_train)

# Make predictions
knn_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, knn_pred)
print("KNN Accuracy:", accuracy)

predicted_labels = encoder.inverse_transform(knn_pred)
print(predicted_labels)
# Saving the trained model
model_filename = "trained_model.pkl"
joblib.dump(knn, model_filename)

app = Flask(__name__)
# Home route
@app.route('/')
def home():
    # Render the input.html page
    return render_template('pg.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        ug_course = int(request.form['ug_course'])
        hsc = int(request.form['hsc'])
        interest = int(request.form['interest'])

        # Make predictions using the loaded model
        new_data = np.array([[ug_course, hsc, interest]])
        prediction = knn.predict(new_data)

        # Map encoded labels back to original class names
        predicted_label = encoder.inverse_transform(prediction)

        # Render the prediction on the stud.html page
        return render_template('stud.html', prediction=predicted_label[0])

    except Exception as e:
        # Handle errors by rendering the input.html page with an error message
        return render_template('pg.html', error=str(e))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
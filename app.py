from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
import os

app = Flask(__name__)

# Load and train model on startup
def train_model():
    """Train the ML model and save it"""
    try:
        # Load data
        df = pd.read_csv('Advertising.csv')
        
        # Remove index column if exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        # Prepare features and target
        X = df[['TV', 'Radio', 'Online']]
        y = df['Sales']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model (better performance)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Save model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print("✓ Model trained and saved successfully!")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

# Load model
if os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    model = train_model()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction based on input"""
    try:
        data = request.json
        
        # Extract inputs
        tv = float(data.get('tv', 0))
        radio = float(data.get('radio', 0))
        online = float(data.get('online', 0))
        
        # Validate inputs
        if tv < 0 or radio < 0 or online < 0:
            return jsonify({'error': 'Values must be non-negative'}), 400
        
        # Make prediction
        if model is not None:
            input_data = np.array([[tv, radio, online]])
            prediction = model.predict(input_data)[0]
            
            return jsonify({
                'success': True,
                'prediction': round(prediction, 2),
                'tv': tv,
                'radio': radio,
                'online': online
            })
        else:
            return jsonify({'error': 'Model not available'}), 500
            
    except ValueError:
        return jsonify({'error': 'Invalid input values'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info')
def info():
    """Get model information"""
    try:
        df = pd.read_csv('Advertising.csv')
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        return jsonify({
            'dataset_rows': len(df),
            'dataset_features': df.shape[1],
            'avg_sales': round(df['Sales'].mean(), 2),
            'max_sales': round(df['Sales'].max(), 2),
            'min_sales': round(df['Sales'].min(), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

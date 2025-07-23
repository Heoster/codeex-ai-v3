#!/usr/bin/env python3
"""
Simple test to check if Flask app starts correctly
"""

from flask import Flask, jsonify

app = Flask(__name__)
app.secret_key = 'test-key'

@app.route('/')
def test_index():
    return jsonify({'status': 'ok', 'message': 'Flask app is working!'})

@app.route('/health')
def health_check():
    return jsonify({'health': 'ok'})

if __name__ == '__main__':
    print("Starting test Flask app...")
    app.run(debug=True, host='127.0.0.1', port=5001)
from flask import Flask, render_template, request, jsonify
import sympy as sm
import re  
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Existing code setup
app = Flask(__name__)
x, y, z = sm.symbols('x y z')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    expression = ''
    operation = ''
    latex_result = ''

    if request.method == 'POST':
        expression = request.form['expression']
        operation = request.form['operation']
        
        # Replace e**x with exp(x) to make sure SymPy interprets it correctly
        expression = re.sub(r'e\*\*([a-zA-Z0-9_]+)', r'exp(\1)', expression)

        # Insert an asterisk between numbers and letters using regex
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)

        try:
            expr = sm.sympify(expression)
            if operation == 'diff':
                result = sm.diff(expr, x)
            elif operation == 'expand':
                result = sm.expand(expr)
            elif operation == 'solve':
                result = sm.solve(expr, x)
            elif operation == 'simplify':
                result = sm.simplify(expr)
            elif operation == 'factor':
                result = sm.factor(expr)
            elif operation == 'rationalize':
                result = sm.together(expr)
            elif operation == 'diffx':
                result = sm.diff(expr, x)
            elif operation == 'diffy':
                result = sm.diff(expr, y)
            elif operation == 'diffz':
                result = sm.diff(expr, z)
            elif operation == 'integratex':
                integral_result = sm.integrate(expr, x)
                result = integral_result + sm.Symbol('C') if integral_result != 0 else sm.Symbol('C')
            elif operation == 'integratey':
                integral_result = sm.integrate(expr, y)
                result = integral_result + sm.Symbol('C') if integral_result != 0 else sm.Symbol('C')
            elif operation == 'integratez':
                integral_result = sm.integrate(expr, z)
                result = integral_result + sm.Symbol('C') if integral_result != 0 else sm.Symbol('C')
            elif operation == 'limit':
                # Assuming limit as x approaches 0, you can modify as needed
                limit_point = float(request.form['limit_point'])
                result = sm.limit(expr, x, limit_point)
            elif operation == 'ode':
                # Solving the ODE expr with respect to t
                result = sm.dsolve(expr, t)
            elif operation == 'series':
                # Compute the series expansion around a point (e.g., x=0)
                expansion_point = float(request.form['expansion_point'])
                result = sm.series(expr, x, expansion_point, n=6)  # n is the number of terms
                
            latex_result = sm.latex(result)
            
        except Exception as e:
            result = f"Error: {str(e)}"
            latex_result = None

    return render_template('index.html', result=result, expression=expression, latex_result=latex_result)

# Chatbot integration

# Define the model class
class ChatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load JSON data
with open('intents.json', 'r') as file:
    data = json.load(file)

# Extract patterns and labels
patterns = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['intent'])

# Tokenize patterns
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Set model parameters
input_size = X.shape[1]
hidden_size = 8
output_size = len(set(labels))

# Load the model
model = ChatModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('Shanks.pth'))
model.eval()

def get_response(message):
    message_vector = vectorizer.transform([message]).toarray()
    message_tensor = torch.tensor(message_vector, dtype=torch.float32)
    output = model(message_tensor)
    _, predicted = torch.max(output, 1)
    intent = label_encoder.inverse_transform(predicted.numpy())[0]

    for intent_data in data['intents']:
        if intent_data['intent'] == intent:
            response = np.random.choice(intent_data['responses'])
            return response

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = get_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=8080, debug=True)


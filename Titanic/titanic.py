import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('Kaggle/train.csv')
test = pd.read_csv('Kaggle/test.csv')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Replace rare titles with 'Other'
data['Title'] = data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 
                                       'Jonkheer', 'Lady', 'Capt', 'Don', 'Mme', 'Mlle', 'Ms'], 'Other')

# Define transformers for different columns
imputer_age = SimpleImputer(strategy='median')
imputer_embarked = SimpleImputer(strategy='most_frequent')
encoder_sex_embarked = LabelEncoder()
scaler = StandardScaler()

# Preprocessing pipeline for numerical features
num_pipeline = Pipeline([
    ('imputer', imputer_age),
    ('std_scaler', scaler),
])

# Define columns for different transformations
num_cols = ['Age', 'Fare']
cat_cols = ['Sex', 'Embarked', 'Title']
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']

# Compose the preprocessing transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', 'passthrough', cat_cols),
        ('drop', 'drop', drop_cols)
    ])

# Apply the transformations
data_preprocessed = preprocessor.fit_transform(data)

# Encode categorical columns
for i in range(3):
    encoder_sex_embarked.fit(data_preprocessed[:, i+2])
    data_preprocessed[:, i+2] = encoder_sex_embarked.transform(data_preprocessed[:, i+2])

# Convert preprocessed data to DataFrame for better visualization
data_preprocessed = pd.DataFrame(data_preprocessed, columns=num_cols+cat_cols)

from sklearn.model_selection import train_test_split

# Define the feature matrix X and target y
X = data_preprocessed.values
y = data['Survived'].values

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 32)  # we now have 10 features, so input_dim = 10
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Create an instance of the model
model = Net()

# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(X_train)
    
    # Compute Loss
    loss = criterion(y_pred, y_train.unsqueeze(1).float())
   
    # Backward pass
    loss.backward()
    optimizer.step()

# Set the model to evaluation mode
model.eval()

# Predict the labels for test data
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = (y_pred > 0.5).float()

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Predict the survival status for the test data
with torch.no_grad():
    y_pred_new = model(X_test_new)
    y_pred_new = (y_pred_new > 0.5).float()

# Create a DataFrame for the output
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred_new.numpy().flatten()})

# Save the output to a CSV file
output.to_csv('predictions.csv', index=False)


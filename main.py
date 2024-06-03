# Import the necessary libraries
# .head(), .tail(), .describe(), and .info() methods can help understand the dataset.
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from evidently import ColumnMapping
# from evidently.dashboard import Dashboard
# from evidently.tabs import DataDriftTab, TargetDriftTab, ModelPerformanceTab
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
import prometheus_client

# Start Prometheus HTTP server
prometheus_client.start_http_server(8000)

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

# Select relevant features and handle missing values
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# add target to the list
titanic = titanic[features + ['Survived']]

# Fill missing values using a dictionary
fill_values = {
    'Age': titanic['Age'].median(),
    'Fare': titanic['Fare'].median(),
    'Embarked': 'S'
}
titanic.fillna(value=fill_values, inplace=True)
# Convert categorical variables to numeric
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
print('Step 1')
# print(titanic.head())
# Prepare features and target
X = titanic[features].values  # each row in one [] row in matrix X
y = titanic['Survived'].values  # one array
print('Step 2')
# print(X)
# print(y)

# Normalize the feature data
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
print('Step 3')
# print(X)

# for using pytorch we have to convert it into tensors
# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
print('Step 4')
print(X_tensor.shape)
print(y_tensor.shape)

# Split into training and testing sets (80/20)
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
#  returns a list of Subset objects.
#  These objects provide indices that refer to a subset of the original dataset
X_train, X_test = torch.utils.data.random_split(X_tensor, [train_size, test_size])
y_train, y_test = torch.utils.data.random_split(y_tensor, [train_size, test_size])

print('Step 5')


#  Function to convert Subset to Tensor
def subset_to_tensor(subset):
    return torch.stack([subset.dataset[i] for i in subset.indices])


# Convert Subset to Tensor for viewing
X_test_tensor = subset_to_tensor(X_test)
y_test_tensor = subset_to_tensor(y_test)

# print("\nX_train Tensor:")
# print(X_test_tensor.shape)
#
# print("\ny_train Tensor:")
# print(y_test_tensor.shape)

# Convert Subset to Tensor for viewing
X_train_tensor = subset_to_tensor(X_train)
y_train_tensor = subset_to_tensor(y_train)

# print("\nX_test Tensor:")
# print(X_train_tensor.shape)
#
# print("\ny_test Tensor:")
# print(y_train_tensor.shape)


# Define a simple MLP model
class TitanicMLP(nn.Module):
    def __init__(self):
        super(TitanicMLP, self).__init__()
        #  Input Layer (fc1) -> Hidden Layer (fc2) -> Output Layer (fc3)
        self.fc1 = nn.Linear(len(features), 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        # print("\nForward Pass:")
        # print(f"Input to fc1: {x[:5]}")
        x = torch.relu(self.fc1(x))
        # print(f"Output from fc1: {x[:5]}")
        x = torch.relu(self.fc2(x))
        # print(f"Output from fc2: {x[:5]}")
        x = self.fc3(x)
        # print(f"Output from fc3 (Final Output): {x[:5]}")
        return x


# Instantiate the model
model = TitanicMLP()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    print(f"\nEpoch [{epoch + 1}/{num_epochs}] - Model Outputs: {outputs.shape}")
    loss = criterion(outputs, y_train_tensor)
    # print(f"Loss: {loss.item()}")

    # Backward pass and optimization
    optimizer.zero_grad()
    #  Compute gradients of the loss with respect to the network's parameters.
    loss.backward()
    print("Gradients after backward pass:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.grad[:5]}")
    #  Adjust the parameters using the computed gradients
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_train_pred = model(X_train_tensor)
    _, predicted_train = torch.max(y_train_pred, 1)
    _, predicted_test = torch.max(y_test_pred, 1)
    # print(y_test_pred.shape)
    accuracy_test = (predicted_test == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy: {accuracy_test:.4f}')
    accuracy_train = (predicted_train == y_train_tensor).sum().item() / len(y_train_tensor)
    print(f'Accuracy: {accuracy_train:.4f}')


# Combine training and testing data for monitoring
print('Combine training and testing data')
train_data = pd.DataFrame(X_train_tensor.numpy(), columns=features)
train_data['prediction'] = predicted_train.numpy()
train_data['target'] = y_train_tensor.numpy()
# print(train_data)
test_data = pd.DataFrame(X_test_tensor.numpy(), columns=features)
test_data['prediction'] = predicted_test.numpy()
test_data['target'] = y_test_tensor.numpy()
# print(test_data)

# Set up column mapping
column_mapping = ColumnMapping(
    target='target',
    prediction='prediction',
    numerical_features=features
)
# Prometheus Metrics
accuracy_metric = prometheus_client.Gauge('model_accuracy', 'Model accuracy on the test set')
precision_metric = prometheus_client.Gauge('model_precision', 'Model precision on the test set')
# recall_metric = prometheus_client.Gauge('model_recall', 'Model recall on the test set')
# f1_metric = prometheus_client.Gauge('model_f1', 'Model F1 score on the test set')
# loss_metric = prometheus_client.Gauge('model_loss', 'Training loss during the last epoch')

# Update Prometheus metrics
accuracy_metric.set(float(accuracy_train))
precision_metric.set(float(accuracy_test))
# recall_metric.set(recall_test)
# f1_metric.set(f1_test)

# Create Evidently Report
print("Generating report...")
report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), ClassificationPreset()])
report.run(reference_data=train_data, current_data=test_data, column_mapping=column_mapping)
report.save_html('model_monitoring_report.html')
print("Report generated and saved.")
print('Prometheus metrics are available at http://localhost:8000')

# Create Evidently Dashboard
# dashboard = Dashboard(tabs=[DataDriftTab(), TargetDriftTab(), ModelPerformanceTab()])
# dashboard.calculate(reference_data=train_data, current_data=test_data, column_mapping=column_mapping)
# dashboard.save_html('model_monitoring_dashboard.html')
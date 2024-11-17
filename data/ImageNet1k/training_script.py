import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--cuda_devices', '-c', type=str, default='4,5,6,7', help='CUDA devices')
argparser.add_argument('--learning_rate', '-l', type=float, default=1e-5, help='Learning rate: lr=1e-i')
argparser.add_argument('--optimizer', '-o', type=str, default='adam', help='Optimizer to use: sgd, adam, amsgrad')
argparser.add_argument('--weight_decay', '-w', type=float, default=1e-10, help='Weight decay: lr=1e-i, then weight_decay=1e-(2i)')
args = argparser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
if not os.path.exists(f'model/{args.learning_rate:.0e}_{args.optimizer}_{args.weight_decay:.0e}'):
    os.makedirs(f'model/{args.learning_rate:.0e}_{args.optimizer}_{args.weight_decay:.0e}')
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    
    def load_state_dict(self, state_dict):
        self.linear.load_state_dict(state_dict)

def compute_loss_and_accuracy(logits, labels):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels.long())
    preds = torch.argmax(logits, 1)
    accuracy = torch.mean((preds == labels).float())
    return loss, accuracy

def compute_top5_accuracy(logits, labels):
    _, top5_preds = torch.topk(logits, 5, dim=1)
    top5_accuracy = torch.mean(torch.sum(top5_preds == labels.unsqueeze(1), dim=1).float())
    return top5_accuracy

def load_data_split(data_type, indices, device):
    features_list = []
    labels_list = []
    for index in indices:
        feature_path = f'image_features/{data_type}/{data_type}_features_{index}.pt'
        features = torch.load(feature_path).to(device)
        features_list.append(features)
        if data_type != 'test':
            label_path = f'image_labels/{data_type}/{data_type}_labels_{index}.pt'
            labels = torch.load(label_path).to(device)
            labels_list.append(labels)
    return features_list, labels_list

def train(model, optimizer, train_indices, val_indices, epochs, devices, patience=10, batch_size=1000):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Split train_indices and val_indices according to the number of devices
    num_devices = len(devices)
    train_splits = [train_indices[i::num_devices] for i in range(num_devices)]
    val_splits = [val_indices[i::num_devices] for i in range(num_devices)]
    
    # Preload data onto each device
    train_data = [load_data_split('train', split, device) for split, device in zip(train_splits, devices)]
    val_data = [load_data_split('validation', split, device) for split, device in zip(val_splits, devices)]
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        total_train_samples = 0
        
        for device, (train_features, train_labels) in zip(devices, train_data):
            for inputs, labels in tqdm(zip(train_features, train_labels)):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(inputs).to(device)
                loss, accuracy = compute_loss_and_accuracy(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                train_accuracy += accuracy.item() * inputs.size(0)
                total_train_samples += inputs.size(0)
        
        train_loss /= total_train_samples
        train_accuracy /= total_train_samples
        val_loss, val_accuracy, val_top5_accuracy = validate(model, val_data, devices)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Top-5 Accuracy: {val_top5_accuracy:.4f}')

        if best_val_loss - val_loss > 1e-3:
            print(f'Validation loss decreased {best_val_loss-val_loss}')
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'Epochs without improvement: {epochs_no_improve}')
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                torch.save(model.state_dict(), f'model/{args.learning_rate:.0e}_{args.optimizer}_{args.weight_decay:.0e}/linear_classifier_epoch_{epoch+1}.pt')
                break

        # save the model every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f'Saving model at epoch {epoch+1} to model/{args.learning_rate:.0e}_{args.optimizer}_{args.weight_decay:.0e}/linear_classifier_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), f'model/{args.learning_rate:.0e}_{args.optimizer}_{args.weight_decay:.0e}/linear_classifier_epoch_{epoch+1}.pt')
        

def validate(model, val_data, devices):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_top5_accuracy = 0.0
    total_val_samples = 0
    
    with torch.no_grad():
        print("validating")
        for device, (val_features, val_labels) in zip(devices, val_data):
            for inputs, labels in tqdm(zip(val_features, val_labels)):
                inputs, labels = inputs.to(device), labels.to(device)
                
                logits = model(inputs).to(device)
                loss, accuracy = compute_loss_and_accuracy(logits, labels)
                top5_accuracy = compute_top5_accuracy(logits, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_accuracy += accuracy.item() * inputs.size(0)
                val_top5_accuracy += top5_accuracy.item() * inputs.size(0)
                total_val_samples += inputs.size(0)
    
    val_loss /= total_val_samples
    val_accuracy /= total_val_samples
    val_top5_accuracy /= total_val_samples
    
    return val_loss, val_accuracy, val_top5_accuracy

def test(model, test_indices, devices, batch_size=1000):
    model.eval()
    num_devices = len(devices)
    test_splits = [test_indices[i::num_devices] for i in range(num_devices)]
    test_data = [load_data_split('test', split, device) for split, device in zip(test_splits, devices)]
    
    with torch.no_grad():
        for device, (test_features, _) in zip(devices, test_data):
            # model.to(device)
            for inputs in test_features:
                inputs = inputs.to(device)
                logits = model(inputs).to(device)
                _, top5_preds = torch.topk(logits, 5, dim=1)
                with open(f'{args.learning_rate:.0e}_{args.optimizer}_{args.weight_decay:.0e}_top5_predictions.txt', 'a') as f:
                    for i in range(top5_preds.size(0)):
                        line = " ".join(map(str, top5_preds[i].tolist()))
                        f.write(f'{line}\n')

# Example usage:
input_dim = 7680
num_classes = 1000
epochs = 100000
devices = [torch.device(f'cuda:{i}') for i in range(len(args.cuda_devices.split(',')))]

train_indices = list(range(1281)) + ['last']
val_indices = list(range(50))
test_indices = list(range(100))

model = LinearClassifier(input_dim, num_classes)
model = nn.DataParallel(model, device_ids=devices)
model.to(devices[0])
if args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == 'amsgrad':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# state_dict = torch.load('model/lre-5_adam/linear_classifier_epoch_120.pt')
# state_dict = {k.replace('module.linear.', ''): v for k, v in state_dict.items()}
# model.load_state_dict(state_dict)
start_time = time.time()
train(model, optimizer, train_indices, val_indices, epochs, devices, patience=10, batch_size=1000)
print(f'Training time: {time.time()-start_time}')
test(model, test_indices, devices, batch_size=1000)
import fire
import numpy as np
import data_processing as dp
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

'''
    Defines the network architecture, activations and regularizers.
    Forward prop.
'''
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


'''
    Sets the loss and optimization criterion and number of epochs.
    They were chosen heuristically.
'''
def set_optimization(model):
    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() 
    # in one single clas
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 5
    return criterion, optimizer, epochs

'''
    forward + backward prop for 1 epoch
    prints the loss for every minibatch (2000 images)
'''
def train_model(model, trainloader, criterion, optimizer, epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[epoch: %d, batch: %5d] loss: %.3f' %
                   (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

'''
    Tests the model accuracy over the test data in one epoch
    Prints the average loss
'''
def test_model(model, testloader, epoch):
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%\n' % (
          100 * correct / total))

'''
    Saves the model to the directory Model
'''
def save_model(net):
    torch.save(net.state_dict(), f="Model/model.model")
    print("Model saved successfully.")

'''
    Loads the pretrained network. 
'''
def load_model(net):
    try:
        net.load_state_dict(torch.load("Model/model.model"))
    except RuntimeError:
        print("Runtime Error!")
        print(("Saved model must have the same network architecture with"
               " the CopyModel.\nRe-train and save again or fix the" 
               " architecture of CopyModel."))
        exit(1) # stop execution with error

'''
    Trains network using GPU, if available. Otherwise uses CPU.
'''
def set_device(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: %s\n" %device)
    # .double() will make sure that  MLP will process tensor
    # of type torch.DoubleTensor:
    return net.to(device), device

'''
    Applies the train_model and test_model functions at each epoch
'''
def train():
    # This loads the dataset and partitions it into batches:
    trainset, testset = dp.load_cifar10()
    trainloader, testloader = dp.batch_data(trainset, testset)
    # Loads the model and the training/testing functions:
    net = SoftmaxClassifier()
    net, _ = set_device(net)
    criterion, optimizer, epochs = set_optimization(net)
    
    # Print the train and test accuracy after every epoch:
    for epoch in range(epochs):
        train_model(net, trainloader, criterion, optimizer, epoch)
        test_model(net, testloader, epoch)

    print('Finished Training')   
    # Save the model:
    save_model(net)

'''
    Classifies the image whose path entered on the terminal.
'''
def test(image_path):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    img_tensor = dp.load_test_image(image_path).unsqueeze(0)
    net = SoftmaxClassifier()
    load_model(net)
    outputs = net(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted: %s" %classes[predicted[0]])


if __name__ == "__main__":
    fire.Fire()


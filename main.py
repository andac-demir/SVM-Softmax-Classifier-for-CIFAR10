import data_processing as dp
from model import SoftmaxClassifier
import model

def save_model(net):
    torch.save(net.state_dict(), f="../TrainedModels/" + 
                              "sofmax_classifier.model")
                                     
def main():
    # This loads the dataset and partitions it into batches:
    trainset, testset = dp.load_cifar10()
    trainloader, testloader = dp.batch_data(trainset, testset)
    # Loads the model and the training/testing functions:
    net = SoftmaxClassifier()
    criterion, optimizer, epochs = model.set_optimization(net)
    
    # Print the train and test accuracy after every epoch:
    for epoch in range(epochs):
        model.train(net, trainloader, criterion, optimizer, epoch)
        model.test(net, testloader, epoch)
    
    # Save the model:
    save_model(net)


if __name__ == "__main__":
    main()
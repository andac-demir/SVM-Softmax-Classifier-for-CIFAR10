import data_processing as dp
from model import SoftmaxClassifier
import model

def main():
    # This loads the dataset and partitions it into batches:
    trainset, testset = dp.load_cifar10()
    trainloader, testloader = dp.batch_data(trainset, testset)
    # Loads the model and the training/testing functions:
    net = SoftmaxClassifier()
    criterion, optimizer, epochs = model.set_optimization(net)
    # TODO: 
    for epoch in epochs:
        model.train(net, trainloader, criterion, optimizer, epoch)
        model.test(net, testloader, epoch)


if __name__ == "__main__":
    main()
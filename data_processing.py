import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), 
                                                         (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                           download=True, transform=transform)
    return trainset, testset

def batch_data(trainset, testset, batch_size=4):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def test_loading(trainloader, classes):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # show images
    imshow(torchvision.utils.make_grid(images))

def main():
    trainset, testset = load_cifar10()
    trainloader, testloader = batch_data(trainset, testset)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    test_loading(trainloader, classes)

if __name__ == '__main__':
    main()




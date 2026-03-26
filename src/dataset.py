from torchvision import datasets, transforms

class Dataset:
    @staticmethod
    def mnist():
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            root="../data/mnist",
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root="../data/mnist",
            train=False,
            download=True,
            transform=transform,
        )
        return train_dataset, test_dataset
        
    @staticmethod
    def fashion_mnist():
        transform = transforms.ToTensor()        
        train_dataset = datasets.FashionMNIST(
            root="../data/fashion_mnist",
            train=True,
            download=True,      
            transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="../data/fashion_mnist",
            train=False,
            download=True,
            transform=transform
        )
        return train_dataset, test_dataset
        
        

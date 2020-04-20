import torchvision.transforms as transforms

CIFAR10_RGB_MEAN = [125.31 / 255.0, 122.95 / 255.0, 113.87 / 255.0]
CIFAR10_RGB_STD = [62.99 / 255.0, 62.09 / 255.0, 66.70 / 255.0]
CIFAR10_SIZE = 32

CIFAR100_RGB_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR100_RGB_STD = [0.2023, 0.1994, 0.2010]
CIFAR100_SIZE = 32

def get_cifar10_transforms():

    transforms_dict = {
        'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),                   
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_RGB_MEAN, CIFAR10_RGB_STD)
            ]),

        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_RGB_MEAN, CIFAR10_RGB_STD)
            ]),

        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_RGB_MEAN, CIFAR10_RGB_STD)
            ])}
    
    return transforms_dict

def get_cifar100_transforms():

    transforms_dict = {
        'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),                   
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_RGB_MEAN, CIFAR100_RGB_STD)
            ]),

        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_RGB_MEAN, CIFAR100_RGB_STD)
            ])}
    
    return transforms_dict
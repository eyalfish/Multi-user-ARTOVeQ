#TODO: run configuration beforehand to load the data
import ssl
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

class DataManager:

    """Singleton class for managing and storing the MNIST dataset.

    This class handles loading, transforming, and splitting the MNIST dataset into
    training, validation, and test sets. It follows the Singleton design pattern
    to ensure that only one instance of this class exists during the application's lifecycle.
    """

    _instance = None

    def __new__(cls) -> 'DataManager':
        """Create or return the single instance of the DataManager class."""
        if cls._instance is None:
            cls._instance = super(DataManager,cls).__new__(cls)
            cls._instance.dataset = cls._instance._generate_dataset()
        return cls._instance

    def _generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates and preprocesses the MNIST dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Load datasets
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

        # Store dataset attributes
        self.train_data, self.train_labels = train_dataset.data, train_dataset.targets
        self.test_data, self.test_labels = test_dataset.data, test_dataset.targets

    def get_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the training and test datasets."""
        return self.train_data, self.train_labels, self.test_data, self.test_labels






















def get_dataloaders():

    #TODO: AUGMENT DATA TO FIT ENCODER MODEL
    if DATASET_NAME == 'CIFAR':
        NUM_CLASSES = 100
        FEATURES = 81920
        def get_test_transforms():
            test_transform = transforms.Compose(
                        [transforms.ToTensor(),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            return test_transform

        def get_train_transforms():
            transform = transforms.Compose(
                        [transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            return transform

        train_transform = get_train_transforms()
        test_transform = get_test_transforms()

        ssl._create_default_https_context = ssl._create_unverified_context
        path = "/tmp/cifar100"
        trainset = datasets.CIFAR100(root = path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR100(root = path, train=False, download=True, transform=test_transform)


        # Define the size of the subset
            # train_subset_size = int(len(trainset) * 0.01)
            # test_subset_size = int(len(testset) * 0.01)
            #
            # # Create random indices for the subset
            # train_indices = np.random.choice(len(trainset), train_subset_size, replace=False)
            # test_indices = np.random.choice(len(testset), test_subset_size, replace=False)
            #
            # # Create the subset
            # train_subset = Subset(trainset, train_indices)
            # test_subset = Subset(testset, test_indices)

        # Create
        #     BATCH_SIZE = 64 DataLoaders for the subsets

        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    elif DATASET_NAME == 'IMAGEWOOF':
        NUM_CLASSES = 10
        FEATURES = 327680
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
        download_url(dataset_url, '.')


        with tarfile.open('./imagewoof2-160.tgz', 'r:gz') as tar: #read file in r mode
          tar.extractall(path = './data') #extract all folders from zip file and store under folder named data

        data_dir = './data/imagewoof2-160'
        # print(os.listdir(data_dir))
        # print(os.listdir('./data/imagewoof2-160/train'))
        # print(len(os.listdir('./data/imagewoof2-160/train')))
        classes = ['Golden retriever', 'Rhodesian ridgeback', 'Australian terrier', 'Samoyed', 'Border terrier', 'Dingo', 'Shih-Tzu', 'Beagle', 'English foxhound', 'Old English sheepdog']

        train_directory = './data/imagewoof2-160/train'
        test_directory = './data/imagewoof2-160/val'

        image_size_test = ImageFolder(train_directory, transforms.ToTensor())

        train_tfms = transforms.Compose([transforms.Resize([SIZE,SIZE]),transforms.ToTensor()])
        test_tfms = transforms.Compose([transforms.Resize([SIZE,SIZE]),transforms.ToTensor()])

        trainset = ImageFolder(data_dir + '/train', train_tfms)
        testset = ImageFolder(data_dir + '/val', test_tfms)

        classes_dict = dict(zip(os.listdir('./data/imagewoof2-160/train'), classes))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    elif DATASET_NAME =='MNIST':
        pass


    return train_dataloader, validation_dataloader, test_dataloader


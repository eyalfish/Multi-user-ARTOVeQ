# Set up all simulation parameters
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, random_split, TensorDataset


from data.data_processor import  DataProcessor



class Config:

    _instance = None # Singleton instance

    def __new__(cls):

        if Config._instance is None:
            Config._instance = object.__new__(cls)
            Config._instance._initialize()
        return Config._instance

    def _initialize(self):
        """Initialize all configuration settings"""
        self.DATASET_NAME = 'MNIST'
        self.SUBSET_SIZE = 20000
        self.BATCH_SIZE = 64
        self.TEST_BATCH_SIZE = 32


        # Device Configuration
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training Hyper-parameters
        self.LEARNING_RATE:float = 3e-4
        self.EPOCHS:int = 10
        self.NUM_USERS:int = 2

        """ Model hyperparameters """

        #Quantizer
        self.QUANTIZER_KWARGS = {'codebook_vector_dim': 2, 'codebook_size': 8}

        # Encoder
        self.ENCODER_KWARGS = {"in_channels": 1, "out_channels": 3}

        # Decoder (Vision Transformer)
        self.DECODER_KWARGS = {
            'image_size': 28,
            'patch_size': 7,
            'num_classes': 10,
            'embed_dim': 512,
            'depth': 2,
            'num_heads': 8,
            'hidden_dim': 10,
            'dropout': 0.2
                                }
        self.FLAG_KWARGS = {'use_quantization': True,
                            'is_training': True}

        self._get_dataloaders()

    def _get_dataloaders(self):
        """Load datasets and create dataloaders."""
        # Get training and test data
        train_data, train_labels = DataProcessor(self.NUM_USERS).get_train_data()
        test_data, test_labels = DataProcessor(self.NUM_USERS).get_test_data()

        # Ensure that the data and labels are in the correct format for TensorDataset
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

        # Split the training data into training and validation sets
        train_set, val_set = random_split(train_dataset, [50000, 10000])

        # Create DataLoaders
        self.TRAIN_LOADER = DataLoader(train_set, batch_size=self.BATCH_SIZE, shuffle=True)
        self.VALIDATION_LOADER = DataLoader(val_set, batch_size=self.BATCH_SIZE, shuffle=False)
        self.TEST_LOADER = DataLoader(test_dataset, batch_size=self.TEST_BATCH_SIZE, shuffle=False)

        # # Check loader lengths for debugging
        # assert(len(self.VALIDATION_LOADER.dataset) == 10000)
        # assert(len(self.TRAIN_LOADER.dataset) == 50000)



def config():
    """Returns the singleton instance of ConfigManager"""
    return Config()


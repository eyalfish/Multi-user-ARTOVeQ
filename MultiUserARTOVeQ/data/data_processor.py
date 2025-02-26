import numpy as np
import torch
from typing import List, Tuple
from data.datamanager import DataManager
from torch.utils.data import Dataset  # Pytorch Dataloader expects the Dataset object

class DataProcessor(Dataset):

    """This class will manage the preprocessing of data and handle splitting, masking,
     concatenation, and any other operations that need to be performed to make the data
     ready for multi-user training."""

    def __init__(self, num_users:int = 2, data_type:str = 'train'):

        self.num_users = num_users
        self.data_manager = DataManager()  # Singleton instance of DataManager

        # Reference to the original source data (training or test)
        self.set_data_type('train')
        if num_users ==2:
            self.get_masks()
        if num_users == 1:
            self.mask1 = torch.ones((28,28))

    def __len__(self):
        return self.source_data.shape[0]

    def __getitem__(self, index:int) ->torch.Tensor:
        """
          Returns the data for all users for a given sample index, using preprocessed data.
          Remark: Applies a mask even if there is a single user

          Args:
              index (int): Index of the sample.

          Returns:
              torch.Tensor: Data for all users at the given index.
          """
        original_image = torch.unsqueeze(self.source_data[index],dim=0) # C,H,W

        users_data = [self.mask_user_data(original_image.clone(),ii+1) for ii in range(self.num_users)]

        return torch.stack(users_data, dim=0)

    def get_masks(self):
        H, W = self.data_manager.train_data[0].shape
        self.mask1 = torch.ones((H,W))
        self.mask1[:H//2,:W//2] = 0
        self.mask2 =  torch.ones((H,W))
        self.mask2[H//2:,W//2:] = 0


    def set_data_type(self, data_type: str):

        """Switches between train and test data dynamically."""

        if data_type == "train":
            self.source_data = self.data_manager.train_data
        elif data_type == "test":
            self.source_data = self.data_manager.test_data
        else:
            raise ValueError("Invalid data type. Expected 'train' or 'test'.")

    def get_train_data(self) -> torch.Tensor:
        """Returns the concatenated training data for all users."""
        self.set_data_type("train")
        x = torch.stack([self.__getitem__(i) for i in range(self.__len__())], dim=0)
        return x, self.data_manager.train_labels

    def get_test_data(self) -> torch.Tensor:
        """Returns the concatenated test data for all users."""
        self.set_data_type("test")

        # Iterate through all sample indices, not just users
        return torch.stack([self.__getitem__(i) for i in range(self.__len__())], dim=0), self.data_manager.test_labels


    def mask_user_data(self, user_data: torch.Tensor, user_id:int) -> torch.Tensor:
        """
        Applies a randomly generated mask to the user's data.

        Args:
            user_data (torch.Tensor): Input image data.

        Returns:
            torch.Tensor: Masked image data.
        """
        if user_id ==1:
            return self.mask1*user_data
        if user_id ==2:
            return self.mask2*user_data



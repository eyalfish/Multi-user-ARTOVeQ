import logging
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import matplotlib.pyplot as plt

import time

# from python_code import DEVICE, conf
# from python_code.channel.channel_dataset import ChannelModelDataset
# from python_code.utils.metrics import calculate_ber
#
# random.seed(conf.seed)
# torch.manual_seed(conf.seed)
# torch.cuda.manual_seed(conf.seed)
# np.random.seed(conf.seed)
class Trainer(object):
    """
     Implements the meta-trainer class. Every trainer must inherit from this base class.
     It initializes the dataloader and model, and provides essential training functionalities.
     """

    def __init__(self, model: nn.Module, learning_rate=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        self._deep_learning_setup()


    def _calculate_loss(self, estimate: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Every trainer must implement its own loss function.
        """
        raise NotImplementedError

    def _forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Defines the forward pass for inference.
        """
        raise NotImplementedError

    def _deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion.
        """
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()


    def run_train_loop(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Implements a single training step.
        """
        predictions = self._forward(inputs)
        loss = self._calculate_loss(predictions, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class Multiuser_ARTOVeQTrainer(Trainer):

    """
    Base training class (meta-trainer) in which other trainers must inheret from this class
    It initializes the dataloader

    """

    def __init__(self, model:nn.Module, learning_rate =1e-3,  device="cuda" if torch.cuda.is_available() else "cpu", train_dataloader=None, validation_dataloader=None ):

        super().__init__(model, learning_rate, device)  # Initializes model, learning rate, and device

        #Initialize datasets (Train and Validation)
        self._initialize_dataloader(train_dataloader,validation_dataloader)


        # Initialize optimizer and loss function
        self._deep_learning_setup()


        # Initialize variable to store previous codebook vectors (useful only for training)
        self.previous_codebook_vectors = None

    def _initialize_dataloader(self, train_dataloader, validation_dataloader):
        if train_dataloader is None or validation_dataloader is None:
            raise ValueError("train_dataloader and validation_dataloader must be provided")
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
    def _deep_learning_setup(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = CrossEntropyLoss()

    def plot_learning_curves(self):

        epochs = range(1, len(self.training_accuracy) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.validation_loss, marker='o', linestyle='-', color='b', label=' Validation Loss')
        plt.plot(epochs, self.training_loss, marker='o', linestyle='-', color='r', label=' Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()



        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.validation_accuracy, marker='o', linestyle='-', color='b', label=' Validation Acc.')
        plt.plot(epochs, self.training_accuracy, marker='o', linestyle='-', color='r', label=' Training Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def _calculate_loss(self, estimate:torch.Tensor, ground_truth:torch.Tensor)->torch.Tensor:
        """
                 Every trainer must have some loss calculation
        """
        return self.criterion(estimate, ground_truth)
    def _calculate_accuracy(self, estimate, ground_truth):

        probs = torch.softmax(estimate.data, dim=1)
        s_hat = probs.argmax(dim=-1)
        return (s_hat == ground_truth).sum()

    def _train(self,max_epochs = 5, num_active_cb_vectors = None):

        """ Runs the training process for the given number of epochs. """


        # Tracking Metrics (Loss, Accuracy)
        self.training_losses = []
        # self.training_accuracy = []
        self.validation_losses = []
        # self.validation_accuracy = []

        start = time.time()
        for epoch in range(max_epochs):
            start = time.time()
            print(f'----- Epoch {epoch + 1}/{max_epochs} -----')

            train_accuracy = [0.0] * int(np.log2(num_active_cb_vectors))
            val_accuracy = [0.0] * int(np.log2(num_active_cb_vectors))
            running_loss = 0.0
            self.model.train()

            for batch_index, (images, labels) in enumerate(self.train_dataloader):

                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                z_decoded, vqvae_loss, current_vectors, z_e = self.model(images,
                                                                         previous_active_vectors=self.previous_codebook_vectors,
                                                                         number_active_vectors=num_active_cb_vectors)  # Multiuser ARTOVEQ has multiple outputs

                loss_levels = []
                # Compute Loss and Accuracy per quantization level
                for q_level in range(len(z_decoded)):
                    cross_entropy_loss = self._calculate_loss(z_decoded[q_level], labels)
                    level_loss = cross_entropy_loss + vqvae_loss[q_level]
                    loss_levels.append(level_loss)

                    train_accuracy[q_level] += self._calculate_accuracy(z_decoded[q_level], labels)

                loss = sum(loss_levels) / len(loss_levels)
                running_loss += loss.item()


                # Backpropagation
                self.optimizer.zero_grad()  # initialize gradients to be zero
                loss.backward()  # Compute gradients
                self.optimizer.step()  # Update network parameters

            # Normalize Loss and Accuracy

            self.training_losses.append(running_loss/len(self.train_dataloader.dataset))

            # Inference loop
            self.model.eval()
            with torch.no_grad():

                running_validation_loss = 0.0
                for batch_index, (test_images, test_labels) in enumerate(self.validation_dataloader):
                    test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)
                    # Forward pass through the model
                    s_hat, vq_vae_validation_loss, _, _ = self.model(test_images,previous_active_vectors=self.previous_codebook_vectors,
                                                                         number_active_vectors=num_active_cb_vectors)
                    level_loss_validation = []

                    for q_level in range(len(s_hat)):
                        CE_validation_loss = self._calculate_loss(s_hat[q_level], test_labels)
                        level_loss_validation.append(CE_validation_loss + vq_vae_validation_loss[q_level])

                        val_accuracy[q_level] += self._calculate_accuracy(s_hat[q_level], test_labels)

                    running_validation_loss+= sum(level_loss_validation).item()
            self.validation_losses.append(running_validation_loss/len(self.validation_dataloader.dataset))


        total_train_accuracy = [100 * (acc.item()) / len(self.train_dataloader.dataset) for acc in train_accuracy]
        total_validation_accuracy =  [100 * (acc.item()) / len(self.validation_dataloader.dataset) for acc in val_accuracy]
        print('\n')
        print(f'Train Accuracy at epoch {epoch + 1} is {total_train_accuracy}%')
        print(f'Train Validation Accuracy at epoch {epoch + 1} is {total_validation_accuracy}%')
        print('---------------------------------')

        encoder_samples = z_e
        # encoder_samples = encoder_samples.permute(0, 2, 3, 1).contiguous()
        # encoder_samples = encoder_samples.view(-1, 2)

        end = time.time()
        print(f' Training loop took {(end-start)/3600} hours')
        return current_vectors, encoder_samples# Return after training for each epoch



    def progressive_training(self, max_quantization_level):


        # Verify that previous_active_vectors is None

        if self.previous_codebook_vectors is not None:
            raise ValueError("No initialization of the previous codebook vectors")

        for level in range(max_quantization_level):

            num_active_cb_vectors = pow(2, level + 1)  # Number of vectors-to-train in CB

            # Run training for this quantization level
            print(f"\nTraining with {num_active_cb_vectors} active codebook vectors (Quantization Level {level + 1})")
            curr_vecs, encoder_samples = self._train(num_active_cb_vectors=num_active_cb_vectors) # training loop for single quantization level

            for train_loss, valid_loss in zip(self.training_losses, self.validation_losses):
                print(f'Training Loss: {train_loss}, Validation Loss: {valid_loss}')

            self.previous_codebook_vectors = curr_vecs.detach() # Update codebook for next level




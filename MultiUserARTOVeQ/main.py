
import logging
from models.decoder import VisionTransformer
from models.multi_user_artoveq import MultiuserARTOVeQ
from utils.config import config
from utils.trainer import Multiuser_ARTOVeQTrainer
from data.data_processor import DataProcessor
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    def _calculate_accuracy( estimate, ground_truth):

        probs = torch.softmax(estimate.data, dim=1)
        s_hat = probs.argmax(dim=-1)
        # print(s_hat, ground_truth)
        pass
        return (s_hat == ground_truth).sum()

    #
    # data_processor = DataProcessor(num_users=1)
    # mnist_source_training_data = data_processor.data_manager.train_data
    # mnist_users_training_data = data_processor.get_train_data() # B,N,H,W
    #
    # batch_index = random.randint(1, data_processor.__len__() - 1) # Select the sample index you want to visualize
    # print(batch_index)
    #
    # num_users = mnist_users_training_data[0].shape[1]  # Number of users
    #
    # # Create a subplot with 1 row and (num_users + 1) columns
    # fig, axes = plt.subplots(1, num_users + 1, figsize=(10, 5))
    #
    # # Plot the original source image
    # axes[0].imshow(mnist_source_training_data[batch_index].detach().numpy(), cmap='gray')
    # axes[0].set_title("Original Image")
    # axes[0].axis("off")
    #
    # # Plot user-specific masked images
    # for i in range(num_users):
    #     axes[i + 1].imshow(mnist_users_training_data[0][batch_index, i].detach().numpy(), cmap='gray')
    #     axes[i + 1].set_title(f"User {i + 1}")
    #     axes[i + 1].axis("off")
    #
    # # Adjust layout and show plot
    # plt.tight_layout()
    # plt.show()
    # print(data_processor.data_manager.train_labels[batch_index])

    cfg = config()
    print('--------------')
    print(cfg.NUM_USERS)
    print('--------------')
    # model = VisionTransformer(**cfg.DECODER_KWARGS,num_users=cfg.NUM_USERS)  # add cfg.NUM_USERS
    model = MultiuserARTOVeQ(cfg.ENCODER_KWARGS, cfg.QUANTIZER_KWARGS, cfg.DECODER_KWARGS, cfg.FLAG_KWARGS)
    model.add_users(cfg.NUM_USERS)


    #TODO: Modify the train loop to be adaptive
    #TODO: Normalize encoder outputs and add rotation trick
    trainer = Multiuser_ARTOVeQTrainer(model, cfg.LEARNING_RATE, cfg.DEVICE, cfg.TRAIN_LOADER, cfg.VALIDATION_LOADER)
    trainer.progressive_training(3)

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    # criterion = nn.CrossEntropyLoss()
    #
    # model.train()
    # for epoch in range(100):
    #
    #     running_loss = 0.0
    #     running_accuracy = 0.0
    #     for batch_index ,(data, labels) in enumerate(cfg.TRAIN_LOADER):
    #
    #         data, labels  = data.to(cfg.DEVICE), labels.to(cfg.DEVICE) # Data does not have the channel dimension
    #         # forward pass
    #         estimates = model(data)
    #         running_accuracy += _calculate_accuracy(estimates,labels)
    #
    #         # Compute Loss
    #         loss = criterion(estimates, labels)
    #
    #         #Backpropagation
    #         optimizer.zero_grad()  # initialize gradients to be zero
    #         loss.backward()  # Compute gradients
    #         optimizer.step()  # Update network parameters
    #
    #
    #         running_loss += loss.item()
    #     print(f'----------------------------')
    #     print(f' epoch {epoch} \t Loss: {running_loss/(cfg.TRAIN_LOADER.dataset.__len__())} \t accuracy: {running_accuracy/(cfg.TRAIN_LOADER.dataset.__len__())}')
    #     print(f'---------------------------- \n')



    # torchsummary.summary(model,(1,28,28))


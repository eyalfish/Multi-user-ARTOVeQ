import torch.nn as nn
import torch


class UserManager(nn.Module):

    """
    Manager class is responsible for handling the creation, storage and forward pass for each encoder
    """
    def __init__(self, Encoder:nn.Module, encoder_kwargs:dict=None):

        super(UserManager,self).__init__()

        # Store the encoder class and arguments for later use
        self.device = self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.encoder = Encoder
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs else {} #TODO:raise error if argument is not provided

        # create a dictionary of user encoders
        self.users_networks = nn.ModuleDict()
        self.user_counter = 0

        # # MNIST DATA
        # self.mnist_data = DataManager() # Singleton instance


    def generate_user_id(self):
        """
              Generate a unique user ID. In this case, a simple counter is used.

              Returns:
                  str: A unique user ID.
        """
        self.user_counter += 1
        return f"user_{self.user_counter}"


    def create_and_add_user(self):
        """
               Create a new user encoder with unique initialization and add it to the manager.

               Returns:
                   str: The user ID of the newly created user.
               """
        user_id = self.generate_user_id()
        new_encoder = self.encoder(**self.encoder_kwargs)
        self.users_networks[user_id] = new_encoder.to(self.device)
        return self


    def __len__(self):
        #TODO: Fix error with the len method
        """
                Return the number of users currently in the manager.

                Returns:
                    int: The number of users in the manager.
        """
        # user network is initialized only after the first nn inserted
        if self.user_counter == 0:
            return 0
        else:
            return len(self.user_networks)

    def forward(self, user_id: str, x: torch.Tensor) :
        #TODO: Perhaps add an input
        return self.users_networks[user_id].forward(x)



#
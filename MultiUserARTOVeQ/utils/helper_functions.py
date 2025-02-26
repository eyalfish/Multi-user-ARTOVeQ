# Helper functions
import torch
def get_accuracy( ground_truth, predicted_values):
    """
            Calculate the accuracy of predictions.

            Args:
                ground_truth (Tensor): Ground truth labels.
                predicted_values (Tensor): Predicted labels.

            Returns:
                int: Number of correct predictions.
    """
    predicted_values = torch.max(predicted_values.data, 1)[1]
    batch_correct = (predicted_values == ground_truth).sum()
    return batch_correct
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os

# from torchinfo import summary


def get_model_configs():
    return {
        "x3d_xs": {
            "side_size": 200,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 200,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 15,
        },
        "x3d_m": {
            "side_size": 278,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 10,
        },
        "x3d_l": {
            "side_size": 340,
            "crop_size": 312,
            "num_frames": 16,
            "sampling_rate": 10,
        },
        "x3d_xl": {
            "side_size": 400,
            "crop_size": 356,
            "num_frames": 16,
            "sampling_rate": 10,
        },
        "x3d_xxl": {
            "side_size": 440,
            "crop_size": 400,
            "num_frames": 16,
            "sampling_rate": 10,
        },
        "swin_t": {
            "side_size": 256,
            "crop_size": 224,
            "num_frames": 16,
            "sampling_rate": 10,
        },
        "mvit_v2_s": {
            "side_size": 256,
            "crop_size": 224,
            "num_frames": 16,
            "sampling_rate": 10,
        },
    }


class ModelBase(ABC):
    def __init__(self, model_name, device, checkpoint_path, classes):
        self.model_name = model_name
        self.frames_per_second = 30
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.class_dict = {idx: name for idx, name in enumerate(classes)}
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            self.load_model_disk(checkpoint_path)
        else:
            self.load_model()

    def get_model_configs(self):
        return get_model_configs()

    def get_std_mean(self):
        return self.mean, self.std

    def cleanup(self):
        pass

    def load_model(self):
        """
        Loads the model, modifies the final layer based on the number of classes,
        and moves it to the specified device.

        - If binary classification (2 classes), uses 1 output node with sigmoid.
        - If multi-class classification (>2 classes), uses N output nodes with softmax.
        """
        # Load pretrained model from PyTorchVideo
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", self.model_name, pretrained=True
        )

        # Get the number of classes
        num_classes = len(self.class_dict)

        # Modify the last layer based on the number of classes
        if num_classes == 2:
            # Binary classification: Use 1 output node + sigmoid activation
            self.model.blocks[-1].proj = nn.Linear(
                self.model.blocks[-1].proj.in_features, 1
            )
            self.activation_fn = nn.Sigmoid()  # Store activation function for inference
        else:
            # Multi-class classification: Use num_classes output nodes + softmax activation
            self.model.blocks[-1].proj = nn.Linear(
                self.model.blocks[-1].proj.in_features, num_classes
            )
            self.activation_fn = nn.Softmax(
                dim=1
            )  # Store activation function for inference

        # Move model to device
        self.model = self.model.to(self.device)

        if not self.model:
            raise ValueError(f"Model is not loaded: {self.model_name}")
        # summary(self.model, input_size=(32, 3, 13, 180, 180))

        print(
            f"Model loaded with {num_classes} output node(s). Using {'sigmoid' if num_classes == 2 else 'softmax'} activation."
        )

    def load_model_disk(self, checkpoint_path):
        """
        Loads the X3D model from a local disk file and adjusts the output layer
        based on the number of classes.

        - If binary classification (2 classes), uses 1 output node with sigmoid.
        - If multi-class classification (>2 classes), uses N output nodes with softmax.
        """
        # Load the model architecture without pre-trained weights
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", self.model_name, pretrained=False
        )

        # Load the checkpoint from the local disk
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract the 'model_state' key if it exists, otherwise use the full checkpoint
        state_dict = checkpoint.get("model_state", checkpoint)

        # Get the number of classes
        num_classes = len(self.class_dict)

        # Modify the last layer based on the number of classes
        if num_classes == 2:
            # Binary classification: 1 output neuron + Sigmoid activation
            self.model.blocks[-1].proj = nn.Linear(
                self.model.blocks[-1].proj.in_features, 1
            )
            self.activation_fn = nn.Sigmoid()  # Store activation function for inference
        else:
            # Multi-class classification: N output neurons + Softmax activation
            self.model.blocks[-1].proj = nn.Linear(
                self.model.blocks[-1].proj.in_features, num_classes
            )
            self.activation_fn = nn.Softmax(
                dim=1
            )  # Store activation function for inference

        # Load the state dictionary into the model
        self.model.load_state_dict(
            state_dict, strict=False
        )  # strict=False allows non-matching keys

        # Move the model to the specified device
        self.model = self.model.to(self.device)
        self.model.eval()

        print(
            f"Model loaded from disk with {num_classes} output node(s). Using {'sigmoid' if num_classes == 2 else 'softmax'} activation."
        )

    @abstractmethod
    def predict_class(self, frames_batches: list):
        """Method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_num_frames(self):
        pass

    @abstractmethod
    def get_sampling_rate(self):
        pass

    @abstractmethod
    def get_validation_transform(self):
        pass

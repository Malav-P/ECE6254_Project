import torch
from typing import Optional, Dict, Tuple
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import matplotlib.pyplot as plt

def feature_extract(model: torch.nn.Module, layers: Dict[str,str], input: torch.tensor) -> Tuple[Dict[str, torch.tensor], torch.fx.GraphModule]:
    """Extracts features from a model.

    Extracts features from specified layers of a model such as ResNet, WideResNet, etc. 

    Args:
        model: A model
        layers: A dictionary of keys and values. Keys represent the layer from the model to 
            extract. Values are the user-specified aliases for these layers. 
        input: A 4D tensor representing a batch of images. 

    Returns:
        A Tuple containing (1) a dictionary of keys and values and (2) a GraphModule object representing the feature extractor. 
        The dictionary keys are the user-specified aliases of the layers. The values are 4D torch tensors of feature maps for each image.
    """

    feature_extractor = create_feature_extractor(model, layers)

    output = feature_extractor(input)
    
    return output, feature_extractor


def get_intermediate_layer(model, layer_to_hook: torch.nn.Module, inputs: torch.tensor) -> torch.tensor:
    """Extracts features from a model.

    Extracts features from specified layers of a model such as ResNet, WideResNet, etc. 

    Args:
        model: A model (torch.nn.Module)
        layer_to_hook: The layer from which to extract feature maps.
        input: A 4D tensor representing a batch of images. 

    Returns:
        A 4D torch tensor of feature maps for each image.
    """

    # Define a hook to access intermediate layers
    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    # Register the hook to the desired layer
    hook_handle = layer_to_hook.register_forward_hook(hook)

    # Set the model to evaluation mode
    model.eval()

    out = model(inputs)

    # Remember to remove the hook after you're done
    hook_handle.remove()

    return outputs[0]

def visualize_features(features: torch.tensor, save_path: Optional[str] = None):
    """Visualize features.

    Plots features and optionally saves feature map to a given directory.

    Args:
        features: A 4D tensor of feature maps.
        save_path: Optionally, a directory to save the feature maps to.
    """

    batch_size = features.shape[0]

    for i in range(batch_size):
        feature_map = features[i]
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]

        gray_scale = gray_scale.detach().numpy()

        fig = plt.figure(figsize=(10, 8))

        a = fig.add_subplot()
        imgplot = plt.imshow(gray_scale)
        a.axis("off")

        if save_path is not None:
            filename = save_path + f"feature_map_img{i}"
            plt.savefig(filename, bbox_inches='tight')

    return 
import torch

def get_intermediate_layer(model, layer_to_hook: torch.nn.Module, inputs: list[torch.tensor]) -> list[torch.tensor]:

    # Define a hook to access intermediate layers
    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    # Register the hook to the desired layer
    hook_handle = layer_to_hook.register_forward_hook(hook)

    # Set the model to evaluation mode
    model.eval()

    for input in inputs:
        output = model(input)

    # Remember to remove the hook after you're done
    hook_handle.remove()

    return outputs
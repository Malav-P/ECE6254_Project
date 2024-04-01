import pytest
from src.FeatureExtraction import get_intermediate_layer

def test_wideresnet():

    import torch
    import torch.nn as nn
    import torchvision.models as models

    # Load pre-trained Wide ResNet-50
    model = models.wide_resnet50_2(weights='DEFAULT')

    input_data = [torch.randn(1,  3, 224, 224)]  # Example input

    with pytest.raises(TypeError):
        output_data = get_intermediate_layer(model=model, layer_to_hook=model.conv1, inputs=None)

    output_data = get_intermediate_layer(model=model, layer_to_hook=model.conv1, inputs=input_data)


    return

def test_resnet():

    import torch
    import torch.nn as nn
    import torchvision.models as models

    # Load pre-trained Wide ResNet-50
    model = models.resnet50(weights='DEFAULT')

    input_data = [torch.randn(1, 3, 224, 224)]  # Example input

    with pytest.raises(TypeError):
        output_data = get_intermediate_layer(model=model, layer_to_hook=model.conv1, inputs=None)

    output_data = get_intermediate_layer(model=model, layer_to_hook=model.conv1, inputs=input_data)


    return

def test_resnext():

    import torch
    import torch.nn as nn
    import torchvision.models as models

    # Load pre-trained Wide ResNet-50
    model = models.resnext50_32x4d(weights='DEFAULT')

    input_data = [torch.randn(1, 3, 224, 224)]  # Example input

    with pytest.raises(TypeError):
        output_data = get_intermediate_layer(model=model, layer_to_hook=model.conv1, inputs=None)

    output_data = get_intermediate_layer(model=model, layer_to_hook=model.conv1, inputs=input_data)


    return

def test_densenet121():

    import torch
    import torch.nn as nn
    import torchvision.models as models

    # Load pre-trained Wide ResNet-50
    model = models.densenet121(weights='DEFAULT')

    input_data = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]  # Example input

    with pytest.raises(TypeError):
        output_data = get_intermediate_layer(model=model, layer_to_hook=model.features.conv0, inputs=None)

    output_data = get_intermediate_layer(model=model, layer_to_hook=model.features.conv0, inputs=input_data)

    assert len(output_data) == 2


    return


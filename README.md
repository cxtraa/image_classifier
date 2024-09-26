## Image Classifier

This is a small web application I built to demonstrate the power of transfer learning.

In this case, I did transfer learning on the [ResNet-50 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html), which is a state-of-the-art convolutional neural network that utilises skip connections. I kept all the weights from the convolutional layers and only trained on the final dense neural network layer mapping to the output logits. The intuition behind this is that ResNet-50 was trained on a much larger dataset of images, so should have learnt features in its earlier layers such as edges and curves, but perhaps not features specialised to CIFAR-100, such as "dog" or "keyboard". The mapping to logits happens in the final layer, so by adjusting that layer only, I can fine-tune the ResNet-50 model for my dataset.

Transfer learning reduces computational costs and is significantly faster to train on, making it an ideal choice if you have a unique dataset but not the time or GPU power to train a model from scratch!
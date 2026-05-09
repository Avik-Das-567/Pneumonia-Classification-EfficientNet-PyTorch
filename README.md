# Pneumonia Classification with EfficientNet-B4 and PyTorch

## Overview

This project builds a deep learning image classifier for detecting pneumonia from pediatric chest X-ray images. The classifier is trained as a binary classification model that assigns each input image to one of two classes:

- `NORMAL`
- `PNEUMONIA`

The implementation uses transfer learning with a pretrained EfficientNet-B4 model from the `timm` library. Instead of training a convolutional neural network from scratch, the notebook freezes the pretrained EfficientNet feature extractor and replaces the original classification layer with a custom fully connected classifier for the two chest X-ray categories.

The main objectives of this project are:

- Apply image transforms for preprocessing and augmentation.
- Load image data into batches using PyTorch `DataLoader`.
- Load and adapt a pretrained EfficientNet model using transfer learning.
- Build a simple PyTorch trainer for model training, validation, checkpointing, and evaluation.

## Problem Statement

Chest X-ray interpretation is a common medical imaging task where visual patterns such as lung opacity, consolidation, and abnormal infiltrates can indicate pneumonia. The goal of this project is to train a neural network that can distinguish between normal chest X-rays and pneumonia-positive chest X-rays.

Formally, the task is supervised binary image classification. Given an input chest X-ray image `x`, the model learns a mapping:

```text
f(x) -> {NORMAL, PNEUMONIA}
```

The notebook trains the model using labeled images stored in class-specific folders and evaluates classification performance using accuracy and cross-entropy loss.

## Dataset

The notebook uses the Kaggle Chest X-Ray Pneumonia dataset:

[Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The dataset is organized into separate `train`, `val`, and `test` folders. Each split contains class subfolders that are read by `torchvision.datasets.ImageFolder`, allowing PyTorch to infer labels directly from the folder names.

The notebook records the following dataset sizes:

| Split | Number of Images |
| --- | ---: |
| Training | 5,216 |
| Validation | 16 |
| Test | 624 |

The class labels used throughout the notebook are:

| Class Index | Class Name |
| ---: | --- |
| 0 | `NORMAL` |
| 1 | `PNEUMONIA` |

## Technical Stack

The project is implemented in Python with the PyTorch ecosystem.

| Component | Usage |
| --- | --- |
| `torch` | Tensor operations, model training, device handling, checkpoint saving, and optimization |
| `torchvision.datasets.ImageFolder` | Loading image datasets from class-labeled folders |
| `torchvision.transforms` | Image resizing, augmentation, tensor conversion, and normalization |
| `torch.utils.data.DataLoader` | Mini-batch creation and dataset shuffling |
| `torchvision.utils.make_grid` | Visualizing a batch of training images as a grid |
| `torch.nn` | Custom classifier head and loss-compatible model layers |
| `torch.nn.functional` | Softmax activation for accuracy and prediction visualization |
| `timm` | Loading the pretrained `tf_efficientnet_b4_ns` model |
| `torchsummary` | Inspecting model architecture, parameter counts, and memory footprint |
| `matplotlib` | Image, grid, and probability visualization |
| `numpy` | Array operations and clipping denormalized image values |
| `tqdm` | Progress bars during training, validation, and testing loops |

The notebook configuration uses:

| Setting | Value |
| --- | --- |
| Epochs | `1` |
| Learning rate | `0.001` |
| Batch size | `16` |
| Input image size | `224 x 224` |
| Model name | `tf_efficientnet_b4_ns` |
| Data directory | `chest_xray_data` |
| Checkpoint file | `PneumoniaModel.pt` |

Device selection is handled dynamically:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

In the recorded notebook run, the model executed on CPU.

## Project Workflow

The project pipeline is organized as follows:

1. Load configuration values such as epochs, learning rate, batch size, model name, image size, and dataset paths.
2. Select the compute device using CUDA when available and CPU otherwise.
3. Define separate image transforms for training, validation, and testing.
4. Load the `train`, `val`, and `test` splits with `torchvision.datasets.ImageFolder`.
5. Wrap each dataset split in a `DataLoader` with a batch size of `16`.
6. Visualize a single transformed image to confirm image loading and label mapping.
7. Visualize a mini-batch using `make_grid` and the custom `show_grid` helper.
8. Load the pretrained EfficientNet-B4 model with `timm.create_model`.
9. Freeze the pretrained backbone parameters by setting `requires_grad = False`.
10. Replace the original classifier with a custom classifier for two output classes.
11. Create the training components: `CrossEntropyLoss`, Adam optimizer, and `PneumoniaTrainer`.
12. Train the model for the configured number of epochs.
13. Track training loss, training accuracy, validation loss, and validation accuracy.
14. Save the model weights to `PneumoniaModel.pt` whenever validation loss improves.
15. Reload the saved checkpoint after training.
16. Evaluate the checkpoint on the test set.
17. Generate prediction probability visualizations for selected test images.

## Image Preprocessing and Augmentation

All images are resized to `224 x 224`, matching the configured input shape used in the notebook. The training split applies a small random rotation to improve robustness to minor orientation changes in X-ray positioning.

Training transform:

```python
T.Compose([
    T.Resize(size=(224, 224)),
    T.RandomRotation(degrees=(-20, +20)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
```

Validation and test transforms:

```python
T.Compose([
    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
```

The normalization values are the standard ImageNet channel statistics. This is appropriate because the EfficientNet backbone is pretrained on ImageNet-style normalized inputs. Although chest X-rays are medical images rather than natural images, matching the pretrained model's expected input distribution helps transfer learning work more effectively.

The helper visualization functions reverse this normalization before displaying images, so plotted examples appear in a human-readable intensity range.

## Model Architecture

The model is based on `tf_efficientnet_b4_ns`, a pretrained EfficientNet-B4 variant loaded through `timm`:

```python
model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
```

EfficientNet uses compound scaling to balance network depth, width, and input resolution. In this project, EfficientNet acts as a feature extractor for chest X-ray images. The pretrained convolutional layers are frozen:

```python
for param in model.parameters():
    param.requires_grad = False
```

The original classifier is replaced with a custom neural network head:

```text
1792 -> 625 -> 256 -> 2
```

The classifier structure is:

```python
model.classifier = nn.Sequential(
    nn.Linear(in_features=1792, out_features=625),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=625, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=2)
)
```

The final layer outputs two logits, one for `NORMAL` and one for `PNEUMONIA`. These logits are passed to `CrossEntropyLoss` during training and converted to probabilities with softmax during prediction visualization.

The recorded model summary reports:

| Parameter Type | Count |
| --- | ---: |
| Total parameters | 18,830,011 |
| Trainable parameters | 1,281,395 |
| Non-trainable parameters | 17,548,616 |

This confirms that the majority of the EfficientNet backbone remains frozen while the new classifier head is trainable.

## Training Strategy

Training is handled by a custom `PneumoniaTrainer` class. The trainer is initialized with a loss function, optimizer, and scheduler placeholder:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
schedular = None
```

The optimizer receives `model.parameters()`, but only parameters with `requires_grad=True` are updated. Since the EfficientNet backbone is frozen, training focuses on the newly added classifier layers.

The trainer contains three main methods:

| Method | Purpose |
| --- | --- |
| `train_batch_loop` | Performs forward pass, loss calculation, backpropagation, optimizer update, and batch-level metric accumulation for the training set |
| `valid_batch_loop` | Runs forward passes on validation or test data and accumulates loss and accuracy |
| `fit` | Coordinates epoch-level training, validation, checkpointing, and metric reporting |

During each training batch:

1. Images and labels are moved to the selected device.
2. The model produces logits.
3. Cross-entropy loss is computed against integer class labels.
4. Gradients are reset with `optimizer.zero_grad()`.
5. Backpropagation is performed with `loss.backward()`.
6. The optimizer updates trainable parameters with `optimizer.step()`.
7. Batch loss and accuracy are accumulated.

Validation runs after each training epoch. If the validation loss improves, the model weights are saved:

```python
torch.save(model.state_dict(), 'PneumoniaModel.pt')
```

This checkpointing strategy keeps the best-performing model according to validation loss.

## Evaluation Results

The notebook trains the model for one epoch and records the following results:

| Metric | Value |
| --- | ---: |
| Training loss | 0.1963 |
| Training accuracy | 0.9271 |
| Validation loss | 0.5690 |
| Validation accuracy | 0.6875 |
| Test loss | 0.2959 |
| Test accuracy | 0.8798 |

The validation split contains only 16 images, so the validation accuracy is based on a very small sample. The test split is larger, with 624 images, and provides a more informative held-out evaluation for the recorded run.

The test evaluation is performed by reloading the saved checkpoint:

```python
model.load_state_dict(torch.load('PneumoniaModel.pt', map_location=device))
model.eval()
avg_test_acc, avg_test_loss = trainer.valid_batch_loop(model, testloader)
```

The final recorded test accuracy is approximately `87.98%`.

## Prediction Visualization

After testing, the notebook visualizes predictions for selected test images. For each selected image:

1. A single test sample is loaded.
2. A batch dimension is added with `unsqueeze(0)`.
3. The model produces logits.
4. Softmax converts logits into class probabilities.
5. `view_classify` displays the X-ray beside a horizontal probability bar chart.

The visualization compares the ground-truth label against the model's predicted class probabilities for:

- `NORMAL`
- `PNEUMONIA`

This makes the final model behavior easier to inspect qualitatively, especially for individual examples where the probability distribution may show high confidence or uncertainty.

## Helper Utilities

The `helper.py` file provides visualization and metric utilities used throughout the notebook.

| Function | Description |
| --- | --- |
| `show_image(image, label, get_denormalize=True)` | Displays a single image tensor with its class label. When denormalization is enabled, it reverses ImageNet normalization before plotting. |
| `show_grid(image, title=None)` | Displays a grid of image tensors, denormalized with ImageNet mean and standard deviation. Used with `torchvision.utils.make_grid`. |
| `accuracy(y_pred, y_true)` | Applies softmax to model logits, selects the highest-probability class with `topk`, compares predictions against labels, and returns mean accuracy. |
| `view_classify(image, ps, label)` | Displays a test image beside a horizontal bar chart of predicted probabilities for `NORMAL` and `PNEUMONIA`. |

The helper functions use the same normalization constants as the image transforms:

```python
mean = torch.FloatTensor([0.485, 0.456, 0.406])
std = torch.FloatTensor([0.229, 0.224, 0.225])
```

This keeps preprocessing and visualization consistent across the project.

## Key Takeaways

- This project demonstrates a complete transfer learning workflow for medical image classification using PyTorch. It combines dataset loading, preprocessing, augmentation, mini-batch training, pretrained EfficientNet feature extraction, classifier replacement, checkpointing, quantitative evaluation, and prediction visualization in a single notebook-driven workflow.
- The final model uses a frozen EfficientNet-B4 backbone and trains a lightweight custom classifier head for binary chest X-ray classification. With one training epoch, the recorded notebook run achieved a test accuracy of approximately `87.98%` on the held-out test split.
- The implementation is intentionally compact but covers the essential components of a supervised computer vision pipeline: data preparation, model adaptation, training loop construction, validation-based checkpointing, and interpretable prediction display.

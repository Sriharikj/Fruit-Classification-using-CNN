Workflow:

1.Data Loading & Augmentation – Load fruit images, normalize, and apply augmentations.

2.Model Setup – Use ResNet50 (pretrained) as base, add custom dense layers with dropout.

3.Training – Compile with Adam optimizer & train using ImageDataGenerator.

4.Evaluation – Plot accuracy/loss curves & validate on test data.

5.Prediction – Classify new fruit images.

6.Deployment – Save trained model in .keras format for reuse.

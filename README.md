# Facial Emotion Recognition with Deep Learning

This repository contains my implementation of a hybrid deep learning model for facial emotion recognition. The model combines local and global feature extraction with attention mechanisms to achieve robust emotion classification.

## Model Architecture

The model architecture is partially inspired by the work of Zhao et al. [1] and uses a hybrid architecture combining:
- ResNet-50 backbone for global feature extraction
- Custom Local Feature Extractor for detailed facial region analysis
- Attention mechanisms at multiple levels
- Integration of local and global features for final classification

## Key Features

- **Local Feature Extraction**: Specialized convolutional layers process different facial regions independently
- **Attention Mechanism**: Multiple attention modules highlight important features at different scales
- **Global-Local Feature Fusion**: Seamless integration of local and global features for better emotion recognition
- **Data Augmentation**: Comprehensive augmentation pipeline for improved model generalization

## Dataset

The model is trained on the RAF-DB dataset, which includes:
- 7 basic emotion categories
- Real-world images with various poses, illuminations, and occlusions
- Balanced training and testing sets

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
pandas>=1.2.3
matplotlib>=3.3.4
seaborn>=0.11.1
scikit-learn>=0.24.1
```

## Results

The model achieves significant performance on the RAF-DB dataset:
- Test Accuracy: ~87%
- Strong performance across all emotion categories
- Particularly robust in distinguishing subtle emotional expressions

## Model Performance

The confusion matrix shows strong performance across all emotion categories:
- High accuracy for "Happy" and "Neutral" emotions
- Good discrimination between similar emotions (e.g., "Sad" vs "Angry")
- Balanced performance across all categories

## Future Improvements

- [ ] Real-time emotion recognition implementation
- [ ] Model compression for mobile deployment
- [ ] Integration with pose estimation for more robust predictions
- [ ] Multi-modal emotion recognition (facial + voice)

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or find any bugs.


## References

[1] Zhao, Z., Liu, Q., & Zhou, F. (2021). Robust Lightweight Facial Expression Recognition Network with Label Distribution Training. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 4, pp. 3510-3519).

```bibtex
@inproceedings{zhao2021robust,
   title={Robust Lightweight Facial Expression Recognition Network with Label Distribution Training},
   author={Zhao, Zengqun and Liu, Qingshan and Zhou, Feng},
   booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
   volume={35},
   number={4},
   pages={3510--3519},
   year={2021}
}
```

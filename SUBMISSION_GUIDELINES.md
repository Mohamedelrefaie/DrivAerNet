
# DrivAerNet++ Submission Guidelines

Thank you for your interest in contributing to the DrivAerNet++ leaderboard! This document outlines the process and requirements for submitting your model's results.

## Submission Process

1. Fork the [DrivAerNet repository](https://github.com/Mohamedelrefaie/DrivAerNet)
2. Evaluate your model using the official train/validation/test splits
3. Create a new branch for your submission
4. Add your results and required files
5. Submit a pull request

## Required Files

Your submission should include:

1. `model_description.md (optional):
   - Model architecture details
   - Implementation specifics
   - Training configuration and hyperparameters
   - Link to paper (if applicable)
   - Link to trained model weights or inference code

2. `test_results.txt`:
   - Complete evaluation metrics on the test set
   - Inference time statistics

## Evaluation Metrics

### 1. Drag Coefficient Prediction

For drag coefficient prediction, the following metrics must be reported:

```python
# Required metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Maximum Absolute Error (Max AE)
- R² Score
- Total inference time and samples processed
```

Example test output format:
```
Test MSE: 0.000123
Test MAE: 0.008976
Max MAE: 0.034567
Test R²: 0.9876
Total inference time: 12.34s for 1200 samples
```

### 2. Surface Field and Volumetric Field Prediction

For surface pressure field and volumetric field predictions, the following metrics must be reported:

```python
# Required metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Maximum Absolute Error (Max AE)
- Relative L1 Error (%) = mean(|prediction - target|_1 / |target|_1)
- Relative L2 Error (%) = mean(|prediction - target|_2 / |target|_2)
- Total inference time and samples processed
```

Example test output format:
```
Test MSE: 0.000456
Test MAE: 0.012345
Max AE: 0.078901
Relative L2 Error: 2.345678
Relative L1 Error: 1.987654
Total inference time: 45.67s for 1200 samples
```

## Code Requirements

### Test Function Implementation

Your evaluation code should follow this structure:

```python
def test_model(model, test_dataloader, config):
    """
    Test the model using the provided test DataLoader and calculate metrics.
    
    Args:
        model: The trained model to be tested
        test_dataloader: DataLoader for the test set
        config: Configuration dictionary containing model settings
    """
    model.eval()
    with torch.no_grad():
        # Implement metric calculations as shown in the example code
        # For drag coefficient:
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)
        r2 = r2_score(all_preds, all_targets)
        
        # For field predictions:
        # Calculate relative errors
        rel_l2 = torch.mean(
            torch.norm(outputs - targets, p=2, dim=-1) / 
            torch.norm(targets, p=2, dim=-1)
        )
        rel_l1 = torch.mean(
            torch.norm(outputs - targets, p=1, dim=-1) / 
            torch.norm(targets, p=1, dim=-1)
        )
```

## Submission Checklist

Before submitting your pull request, ensure:

- [ ] All required metrics are calculated and reported
- [ ] Results are obtained using the official data splits
- [ ] Model description is complete and clear
- [ ] Code follows the provided format for metric calculation
- [ ] All results are reproducible

## Review Process

1. Your submission will be reviewed for completeness
2. Results will be verified for correctness
3. Upon approval, your results will be added to the leaderboard

For questions or clarifications, please contact:
Mohamed Elrefaie (email: mohamed.elrefaie [at] mit [dot] edu)

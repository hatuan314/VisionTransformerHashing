# Prompt Scenario 2: Backbone Comparison with Fixed CSQ Framework

## Context
This scenario compares different backbone architectures (AlexNet, ResNet, ViT-B_16, ViT-B_32) using the same hashing framework (CSQ) on CIFAR-10 with 32-bit hash codes.

## Question for AI Agent
Based on the training logs data below, provide visualization recommendations for comparing backbone architectures in image hashing:

1. **Which backbones provide the best performance** and how significant are the differences?
2. **How should we visualize the convergence patterns** of different backbone architectures?
3. **What visualization approach best shows** the speed vs. accuracy trade-offs?
4. **Recommend chart strategies** to demonstrate why ViT backbones outperform traditional CNNs.

## Data Summary
- **Framework**: CSQ (Consistency Sensitive Quantization)
- **Dataset**: CIFAR-10
- **Hash Bits**: 32
- **Backbones Tested**: AlexNet, ResNet, ViT-B_16, ViT-B_32
- **Training Duration**: 150 epochs with evaluation at epochs 30, 60, 90, 120
- **Key Metrics**: Training loss per backbone, Test MAP, Precision-Recall curves

## Backbone Performance Results
```
- AlexNet (CSQ, 32-bit):    Best MAP = 0.843
- ResNet (CSQ, 32-bit):     Best MAP = 0.843
- ViT-B_16 (CSQ, 32-bit):   Best MAP = 0.966
- ViT-B_32 (CSQ, 32-bit):   Best MAP = 0.954
```

## Data File
Detailed training logs and metrics: `assets/scenario_logs/scenario-002/summary.json`

## Expected Output
Recommendations for:
- Comparative visualization of CNN vs Transformer backbones
- How to show the performance gap clearly
- Convergence speed comparison charts
- Visual hierarchy for emphasizing key findings

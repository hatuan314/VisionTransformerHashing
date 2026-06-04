# Prompt Scenario 1: Framework Comparison with Fixed ViT-B_32 Backbone

## Context
This scenario compares different hashing frameworks (CSQ, IDHN, DPN, HashNet) using the same backbone (Vision Transformer ViT-B_32) and dataset (CIFAR-10) with 32-bit hash codes.

## Question for AI Agent
Based on the training logs data below, please provide visualization recommendations for comparing the performance of different hashing frameworks on CIFAR-10 with ViT-B_32 backbone:

1. **Which frameworks perform best** in terms of final MAP (Mean Average Precision)?
2. **How should we visualize the convergence behavior** across frameworks during training?
3. **What chart types would best show** the trade-offs between different frameworks?
4. **Suggest optimal visualization strategies** for a research paper presentation comparing these 4 frameworks.

## Data Summary
- **Dataset**: CIFAR-10
- **Backbone**: Vision Transformer (ViT-B_32)
- **Hash Bits**: 32
- **Frameworks Tested**: CSQ, IDHN, DPN, HashNet
- **Training Duration**: 150 epochs with evaluation at epochs 30, 60, 90, 120
- **Key Metrics**: Training loss, Precision-Recall curve, Test MAP scores

## Framework Performance Results
```
- CSQ (ViT-B_32, 32-bit):    Best MAP = 0.954
- IDHN (ViT-B_32, 32-bit):   Best MAP = 0.959
- DPN (ViT-B_32, 32-bit):    Best MAP = 0.958
- HashNet (ViT-B_32, 32-bit): Best MAP = 0.952
```

## Data File
Detailed training logs and metrics: `assets/scenario_logs/scenario-001/summary.json`

## Expected Output
Please recommend:
- Line charts, bar charts, or other visualization types for comparing frameworks
- How to highlight the best-performing method
- Metrics to emphasize (MAP, convergence speed, stability)
- Visual design recommendations for clarity and publication quality

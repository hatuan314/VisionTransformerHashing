# Prompt Scenario 3: All Tokens vs CLS Token Representation

## Context
This scenario evaluates the impact of using only CLS token (cls variants) versus all tokens in Vision Transformer-based hashing on CIFAR-10 with 32-bit hash codes.

## Question for AI Agent
Based on the training logs data below, provide visualization recommendations for showing the impact of token representation choices in ViT-based hashing:

1. **How much does using only CLS token impact performance?** (identify the MAP difference)
2. **Which visualization approach best shows** this trade-off between all-tokens vs CLS-token methods?
3. **How should we visualize convergence differences** between the two approaches?
4. **Recommend visual strategies** to clearly communicate this finding to readers.

## Data Summary
- **Backbone**: Vision Transformer (ViT-B_32)
- **Dataset**: CIFAR-10
- **Hash Bits**: 32
- **Frameworks Compared**: All-tokens (DSH, CSQ, DPN, HashNet) vs CLS-token variants (DSHcls, CSQcls, DPNcls, HashNetcls)
- **Training Duration**: 150 epochs with evaluation at epochs 30, 60, 90, 120
- **Key Metrics**: Training loss, Test MAP, Precision-Recall curves

## Performance Comparison (All Tokens vs CLS Token)
```
All Tokens Variants:
- DSH (all tokens):       Best MAP = 0.939
- CSQ (all tokens):       Best MAP = 0.954
- DPN (all tokens):       Best MAP = 0.958
- HashNet (all tokens):   Best MAP = 0.952

CLS Token Variants:
- DSHcls:                 Best MAP = 0.948
- CSQcls:                 Best MAP = 0.944
- DPNcls:                 Best MAP = 0.951
- HashNetcls:             Best MAP = 0.953
```

## Data File
Detailed training logs and metrics: `assets/scenario_logs/scenario-003/summary.json`

## Expected Output
Recommendations for:
- Side-by-side visualization strategies (paired comparisons)
- How to highlight the subtle MAP differences
- Convergence behavior comparison charts
- Visual design for showing this is a nuanced finding (not a dramatic difference)

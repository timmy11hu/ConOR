# ConOR
[ECCV 2022] Uncertainty Quantification in Depth Estimation via Constrained Ordinal Regression

## Simulation Experiments

### Dataset 1 (error variance)

To examine error variance estimation of ConOR, run 

```python main.py --cls_model conor --sample uniform --method Single```

### Dataset 2 (estimation variance)

To examine estimation of esimation variance with boostrap on Conor, first get ground truth by running 

```python main.py --cls_model conor --sample gaussian --method Truth --num_ensemble 100```

Then run boostrapping methods: 

 ```python main.py --cls_model conor --sample gaussian --method MultBS --num_ensemble 100```





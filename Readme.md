# VGMGC

This is the code of paper: Variational Graph Generator for Multi-View Graph Clustering

# Requirements

- Python 3.8
- Pytorch 1.11.0
- munkres 1.1.4
- scikit-learn 1.0.1
- scipy 1.8.0



# Datasets

ACM and DBLP are included in `./data/`. The other datasets are public available. 

|     Dataset      | #Clusters | #Nodes |   #Features    |                            Graphs                            |
| :--------------: | :-------: | :----: | :------------: | :----------------------------------------------------------: |
|       ACM        |     3     |  3025  |      1830      |   $\mathcal{G}^1$ co-paper<br />$\mathcal{G}^2$ co-subject   |
|       DBLP       |     4     |  4057  |      334       | $\mathcal{G}^1$ co-author<br />$\mathcal{G}^2$ co-conference<br />$\mathcal{G}^3$ co-term |
|  Amazon photos   |     8     |  7487  | 745<br />7487  |                 $\mathcal{G}^1$ co-purchase                  |
| Amazon computers |    10     | 13381  | 767<br />13381 |                 $\mathcal{G}^1$ co-purchase                  |

# Test VGMGC

```python
# Test VGMGC on ACM dataset
python vgmgc.py --dataset 'acm' --train False --model_name 'vgmgc_acm.pkl' --order 8 --lam_emd 1

# Test VGMGC on DBLP dataset
python vgmgc.py --dataset 'dblp' --train False --model_name 'vgmgc_dblp.pkl' --order 8 --lam_emd 5
```

# Train VGMGC

```python
# Train VGMGC on ACM dataset
python vgmgc.py --dataset 'acm' --train True --model_name 'vgmgc_acm1.pkl' --order 8 --weight_soft 0.9 --min_belief 0.7 --max_belief 0.99 --lam_emd 1 --kl_step 5 --lam_elbo_kl 1 --threshold 0.8 --temperature 5

# Train VGMGC on DBLP dataset
python vgmgc.py --dataset 'dblp' --train True --model_name 'vgmgc_dblp1.pkl' --order 8 --weight_soft 0.1 --min_belief 0.2 --max_belief 0.99 --lam_emd 5 --kl_step 10 --lam_elbo_kl 1 --threshold 0.8 --temperature 1
```

**Parameters**: More parameters and descriptions can be found in the script and paper.

# Results of VGMGC

|                  | NMI% | ARI% | ACC% | F1%  |
| :--------------: | :--: | :--: | :--: | :--: |
|       ACM        | 77.3 | 83.7 | 94.3 | 94.3 |
|       DBLP       | 78.3 | 83.7 | 93.2 | 92.7 |
|  Amazon photos   | 66.8 | 58.4 | 78.5 | 76.9 |
| Amazon computers | 53.5 | 47.5 | 62.2 | 50.2 |

# VGMGC

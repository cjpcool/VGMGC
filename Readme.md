from vgmgc import cuda_devicefrom torch.distributed.pipeline.sync.stream import use_devicefrom vgmgc import update_intervalfrom vgmgc import latent_dim

# VGMGC

This is the code of paper: Variational Graph Generator for Multi-View Graph Clustering.


We sincerely appreciate it if you cite this paper as: 
~~~
@ARTICLE{10833915,
  author={Chen, Jianpeng and Ling, Yawen and Xu, Jie and Ren, Yazhou and Huang, Shudong and Pu, Xiaorong and Hao, Zhifeng and Yu, Philip S. and He, Lifang},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Variational Graph Generator for Multiview Graph Clustering}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2024.3524205}}

~~~


# Requirements

- Python >= 3.8
- Pytorch >= 1.11.0
- munkres >= 1.1.4
- scikit-learn >= 1.0.1
- scipy >= 1.8.0



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
python vgmgc.py --dataset acm --train False --model_name vgmgc_acm.pkl --order 8 --lam_emd 1

# Test VGMGC on DBLP dataset
python vgmgc.py --dataset dblp --train False --model_name vgmgc_dblp.pkl --order 8 --lam_emd 5

# Test VGMGC on Cora dataset
python vgmgc.py --dataset cora --train False --model_name vgmgc_cora.pkl --order 10 --weight_soft 1. --min_belief 0.2 --max_belief 0.99 --lam_emd 0.2 --kl_step 5 --lam_elbo_kl 1 --threshold 0.5 --temperature 1 --add_graph True 

# Test VGMGC on Citeseer dataset
python vgmgc.py --dataset citeseer --train False --model_name vgmgc_citeseer.pkl --order 8 --weight_soft 1. --min_belief 0.2 --max_belief 0.99 --lam_emd 1. --kl_step 5 --lam_elbo_kl 1 --threshold 0.5 --temperature 1 --add_graph True

# Test VGMGC on 3Sources dataset
python vgmgc.py --dataset 3sources --train False --model_name vgmgc_3sources_acc0.9467.pkl --order 1 --weight_soft 1. --min_belief 0.2 --max_belief 0.99 --lam_emd 10. --kl_step 5 --lam_elbo_kl 1 --threshold 0.5 --temperature 1

# Test VGMGC on bbc sport  dataset
python vgmgc.py --dataset bbcsport_2view --train False --model_name vgmgc_bbcsport_2view_acc0.9835.pkl --order 2 --weight_soft 1. --min_belief 0.2 --max_belief 0.99 --lam_emd 100. --kl_step 5 --lam_elbo_kl 1 --threshold 0.5 --temperature 1


```
# Train VGMGC

```python
# Train VGMGC on ACM dataset
python vgmgc.py --dataset acm --train true --model_name vgmgc_acm1.pkl --order 8 --weight_soft 0.9 --min_belief 0.7 --max_belief 0.99 --lam_emd 1 --kl_step 5 --lam_elbo_kl 1 --threshold 0.8 --temperature 5

# Train VGMGC on DBLP dataset
python vgmgc.py --dataset dblp --train true --model_name vgmgc_dblp1.pkl --order 8 --weight_soft 0.1 --min_belief 0.2 --max_belief 0.99 --lam_emd 5 --kl_step 10 --lam_elbo_kl 1 --threshold 0.8 --temperature 1


python vgmgc.py --dataset cora --train true --model_name vgmgc_cora1.pkl --order 10 --weight_soft 1. --min_belief 0.2 --max_belief 0.99 --lam_emd 0.2 --kl_step 5 --lam_elbo_kl 1 --threshold 0.5 --temperature 1  --add_graph True --update_interval 2


python vgmgc.py --dataset citeseer --train true --model_name vgmgc_citeseer1.pkl --order 8 --weight_soft 1. --min_belief 0.2 --max_belief 0.99 --lam_emd 1. --kl_step 5 --lam_elbo_kl 1 --threshold 0.5 --temperature 1  --add_graph True --update_interval 2


python vgmgc.py --dataset 3sources --train true  --order 1 --weight_soft 1. --min_belief 0.2 --max_belief 0.99 --lam_emd 10. --kl_step 5 --lam_elbo_kl 1 --threshold 0.5 --temperature 1 --latent_dim 512 --hidden_dim 512 --update_interval 2


python vgmgc.py --dataset bbcsport_2view --train true  --order 2 --weight_soft 1. --min_belief 0.2 --max_belief 0.99 --lam_emd 100. --kl_step 5 --lam_elbo_kl 1 --threshold 0.5 --temperature 1 --latent_dim 512 --hidden_dim 512 --update_interval 2

```

**Parameters**: More parameters and descriptions can be found in the script and paper.

# Results of VGMGC

|                  | NMI% | ARI% | ACC% | F1%  |
| :--------------: | :--: | :--: | :--: | :--: |
|       ACM        | 77.3 | 83.7 | 94.3 | 94.3 |
|       DBLP       | 78.3 | 83.7 | 93.2 | 92.7 |
|  Amazon photos   | 66.8 | 58.4 | 78.5 | 76.9 |
| Amazon computers | 53.5 | 47.5 | 62.2 | 50.2 |


|           | NMI% | ARI% | ACC% | F1%  |
|:---------:|:----:|:----:|:----:|:----:|
|   Cora    | 55.6 | 51.7 | 73.5 | 71.6 |
| Citeseer  | 45.2 | 46.4 | 70.1 | 65.4 |
| BBC sport | 94.6 | 95.1 | 98.3 | 98.6 |
| 3sources  | 86.5 | 88.1 | 94.7 | 93.5 |


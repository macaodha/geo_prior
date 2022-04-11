# Presence-Only Geographical Priors for Fine-Grained Image Classification
Code for recreating the results in our ICCV 2019 [paper](https://arxiv.org/abs/1906.05272).  

`demo.py` is a simple demo script that either 1) takes location as input and returns a prediction for all the categories predicted to be present at that location or 2) generates a dense prediction for a category of interest.  
`geo_prior/` contains the main code for training and evaluating models.  
`gen_figs/` contains scripts to recreate the plots in the paper.  
`pre_process/` contains scripts for training image classifiers and saving features/predictions.  
`web_app/` contains code for running a web based visualization of the model predictions.   


### Example Predictions
For more results, data, and an interactive demo please consult our project [website](https://homepages.inf.ed.ac.uk/omacaod/projects/geopriors/index.html).
<p align="center">
  <img src="data/example_predictions.jpg" alt="example_predictions" width="1000" />
</p>


### Reference
If you find our work useful in your research please consider citing our paper.  
```
@inproceedings{geo_priors_iccv19,
  title     = {{Presence-Only Geographical Priors for Fine-Grained Image Classification}},
  author    = {Mac Aodha, Oisin and Cole, Elijah and Perona, Pietro},
  booktitle = {ICCV},
  year = {2019}
}
```

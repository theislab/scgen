# Generative modeling and latent space arithmetics predict single-cell perturbation response across cell types, studies and species.

<img align="center"  src="/sketch/sketch.png?raw=true">



This repository includes python scripts and notebooks in code folder to reproduce figures from the paper [(bioRxiv, 2017)](https://www.biorxiv.org/content/10.1101/478503v2) according to the table bellow.

figure       | notebook/script     
---------------| ---------------
| 2, s1, s2, s3, s4, s7  | scgen_kang, cvae, st_gan, vec_arith_pca, vec_arith, scgen_sal_ta | 
|        3          | scgen_hpoly, scgen_salmonella| 
|        4          | cross_study| 
|        5, s8      | cross_species|
|        6, s9      | pancrease, bbknn, mnn, cca, scanorama|
|        s6      |scgen_kang_multiple|
|        s10        |mouse_atlas| 

To run the notebooks and scripts you need follwoing packages :


tensorflow (1.4), scanpy, numpy, matplotlib, scipy, wget.

Note: The Current contents of this repo will be moved to a new repository in near future. That repository will include codes for reproducing the results of the paper and the current one will only include the software.



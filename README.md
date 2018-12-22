# Generative modeling and latent space arithmetics predict single-cell perturbation response across cell types, studies and species.

<img align="center"  src="/sketch/sketch.png?raw=true">

This repository includes python scripts and notebooks in code folder to reproduce figures from the paper based on the table bellow.

figure       | notebook/script     
---------------| ---------------
| 2, s1, s2, s3, s7  | scgen_kang, cvae, st_gan, vec_arith_pca, vec_arith, scgen_sal_TA | 
|        3          | scgen_hpoly, scgen_salmonella| 
|        4          | cross_study| 
|        5, s8      | cross_species|
|        6, s9      | pancrease, bbknn, mnn, cca, scanorama|
|        s6      |scgen_kang_multiple|
|        s10        |mouse_atlas| 

To run the notebooks and scripts you need follwoing packages :
tensorflow (1.4), scanpy, numpy, matplotlib, scipy, wget.

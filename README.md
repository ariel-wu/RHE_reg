# RHE_reg
Randomized HE regression estimator 


We propose a scalable randomized Method-of-Moments (MoM) estimator of SNP heritability and genetic correlations in LMMs. Our method, RHE-reg, leverages the structure of genotype data to obtain runtimes that are sub-linear in the number of individuals (assuming the number of SNPs is held constant).

### Citing RHE_reg

If you find that this software is useful for your research project,
please cite our paper:

Wu Y.,Sankararaman S. (2018) A scalable estimator of SNP heritability for Biobank-scale Data

## Getting Started

### Prerequisites
The following packages are required on a linux machine to compile and use the software package. 
```
g++
cmake
make
```

### Installing
Installing RHE_reg is fairly simple. Follow the following steps: 
```
git clone https://github.com/ariel-wu/RHE_reg.git
cd RHE_reg
mkdir build 
cd build
cmake .. 
make
```

## Documentation for RHE_reg

After compiling the executble fastppca is present in the build directory. 
To run RHE_reg, use

*``./RHE_reg <command_line arguments> ``

### Parameters

The values in the brackets are the command line flags for running the code without parameter file. 

```
* genotype (-g) : The path of the genotype file or plink binary file prefix.
* phenotype (-p) : The path of the phenotype file. 
* covariate (-c) : The path of the covariate file.
* covariate name (-cn) : The name of the covariate to use in the covariate file. If not specified, RHE_reg will use all the covarites in the covariate file. 
* batch number (-b) : Number of random vectors used in the estimator % 10. 
* phenotype number (-mpheno) : The number of phenotype to use in the phenotype file. If not specified, RHE_reg will compute heritability estimates on all the phenotypes. 
* fill in missing phenotype with mean (-fill) : Fill in missing phenotypes with mean. Otherwise will be ignored. 
```


An example parameter file is provided in the example directory. 
You can run the code using the command: 

```
../build/RHE_reg -g 200_100 -p 200_100.pheno.plink -c 200_100.cov -b 10 
```



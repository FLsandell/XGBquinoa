
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/FLSandell/XGQuinoa">
    <img src="images/BOKU-Logo-150-Institut-ICB-kl.png" alt="Logo" width="138" height="45">
  </a>

<h3 align="center">XGBquinoa</h3>

  <p align="center">
    Unrevealing the genomic basis of seed colour using Extreme Gradient Boosting
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

Quinoa is an agriculturally important crop species originally domesticated in the Andes of central South America. One of its most important phenotypic traits is seed colour. Seed colour variation is determined by contrasting abundance of betalains, a class of strong antioxidant and free radicals scavenging colour pigments only found in plants of the order Caryophyllales. However, the genetic basis for these pigments in seeds remains to be identified. Here we demonstrate the application of machine learning (extreme gradient boosting) to identify genetic variants predictive of seed colour. We show that machine learning outperforms the classical genome-wide association (GWAS) approach. We provide re-sequencing and phenotypic data for 156 South American quinoa accessions and identify candidate genes potentially controlling betalain content in quinoa seeds. Genes identified include novel cytochrome P450 genes and known members of the betalain synthesis pathway, as well as genes annotated as being involved in seed development. Our work showcases the power of modern machine learning methods to extract biologically meaningful information from large sequencing data sets.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The code is available as Jupyter notebook (Colours_Quinoa_Code.ipynb) and as text file (Colours_Quinoa_Code.py). The code was designed to predict seed color in quinoa based on sequencing data. The input file is a 0|1|2 matrix, where 0 denotes a homozygous reference position, 1 a heterozygous alternative position, and 2 a homozygous alternative position in the genome assembly of the Bolivian reference genotype CHEN125. The variant matrix for the sequencing dataset comprising 156 quinoa accessions is available at [10.6084/m9.figshare.24466522](https://doi.org/10.6084/m9.figshare.24466522.v1 ). The seed colors are listed in Quinoa_colour_info.txt (see InputFiles).

The code can readily be modified to construct a predictive model for any other trait that can be inferred from sequencing data. To generate a similar input matrix for new data, a VCF file can be converted by using vcftools with the "--012" flag.

### Prerequisites

python3

jupyter (optional, to view Colours_Quinoa_Code.ipynb)

The following python modules:

xgboost

pandas

numpy

hyperopt

sklearn

matplotlib

<!-- Information about our group -->
## About the ICB

If you are interested in our work you can find more information [here](https://bvseq.boku.ac.at/) and on our [twitter](https://twitter.com/ICBboku).


<!-- LICENSE -->
## License

Copyright (c) 2023 Felix Sandell

Distributed under the MIT License. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Felix Leopold Sandell - [@flwasch](https://twitter.com/flwasch) - felix.sandell@boku.ac.at


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">XGQuinoa</h3>

  <p align="center">
    Unrevealing the genomic basis of seed colour using Extreme Gradient Boosting
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

Quinoa is an agriculturally important crop species originally domesticated in the Andes of central South America. One of its most important phenotypic traits is seed colour. Seed colour variation is determined by contrasting abundance of betalains, a class of strong antioxidant and free radicals scavenging colour pigments only found in plants of the order Caryophyllales. However, the genetic basis for these pigments in seeds remains to be identified. Here we demonstrate the application of machine learning (extreme gradient boosting) to identify genetic variants predictive of seed colour. We provide re-sequencing and phenotypic data for 156 South American quinoa accessions and identify candidate genes potentially controlling betalain content in quinoa seeds. Genes identified include novel cytochrome P450 genes and known members of the betalain synthesis pathway, as well as genes annotated as being involved in seed development. Our work showcases the power of modern machine learning methods to extract biologically meaningful information from large sequencing data sets.

(insert link to final publication)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The code associated with this publication is available as a Jupyter notebook. While the code was originally designed to forecast seed color based on quinoa sequencing data, it can readily be modified to construct a predictive model for any trait that can be inferred from sequencing data. The input file should be a 0|1|2 matrix, where 0 denotes a homozygous reference variant, 1 signifies a heterozygous variant, and 2 indicates a homozygous alternative variant. We have provided this matrix for the sequencing dataset comprising 156 accessions, as mentioned in this publication. To generate a similar matrix for new data, a VCF file can be employed, and the conversion can be facilitated using vcftools with the "--012" flag.

### Prerequisites

python3

jupyter 

The Jupyter notebook displays all the required python modules.

<!-- Information about our group -->
## Roadmap

If you are interested in our work you can find more informations [here](https://bvseq.boku.ac.at/) and on our [twitter](https://twitter.com/ICBboku).


<!-- Cite -->
## How to cite us

When utilizing this provided code in any manner, please ensure to cite the following research paper:

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@flwasch](https://twitter.com/flwasch) - felix.sandell@boku.ac.at


<p align="right">(<a href="#readme-top">back to top</a>)</p>


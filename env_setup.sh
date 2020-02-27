export https_proxy=http://proxy-internet-aws-eu.subsidia.org:3128
export http_proxy=http://proxy-internet-aws-eu.subsidia.org:3128
export no_proxy=169.254.169.254,127.0.0.1
git config --global credential.helper store
git config --global user.email "benjamin.bouillet@decathlon.com"
git config --global user.name "Ben Bouillet"
conda env create environment.yml
source activate forecast-modeling-demand
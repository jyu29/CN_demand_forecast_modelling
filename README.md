<p align="center">
  <img src="https://user-images.githubusercontent.com/15980664/101493322-3abf6000-3966-11eb-9e23-c902b2109e13.png" data-canonical-src="https://user-images.githubusercontent.com/15980664/101493322-3abf6000-3966-11eb-9e23-c902b2109e13.png" width="500"/>
</p>

<p align="center">
  <a href="https://https://github.com/dktunited/forecast-modeling-demand/releases/" target="_blank">
    <img alt="GitHub release" src="https://img.shields.io/github/v/release/dktunited/forecast-modeling-demand?include_prereleases&style=flat-square">
  </a>
  
  <a href="https://https://github.com/dktunited/forecast-modeling-demand#contribute" target="_blank">
    <img alt="Contributors" src="https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square">
  </a>
</p>

# Decathlon Demand Forecast - Forecast United Modeling

A Sagemaker-backed, DeepAR-based, machine learning application to forecast sales for the Decathlon Demand Team.

Curated with :heart: by the Forecast United team.

Model used - [**DeepAR**](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html) developed by **AWS**.

Reach out to us! Join the [**Slack channel**](https://join.slack.com/t/forecastunited/shared_invite/zt-jyntaf4k-j6cX_73RwBLr4DR9dN0PwQ).

## Table of contents

- [Usage](##usage)
- [Development](##development)
- [Architecture](##Architecture)
- [Contribute](##contribute)

## Usage

* Clone the repository
```sh
git clone https://github.com/dktunited/forecast-modeling-demand/
cd forecast-modeling-demand/
```

* Install & activate the conda environment
```sh
conda env create -f environment.yml
conda activate forecast-modeling-demand
```

* IAM Authentication : as per this [wiki](https://wiki.decathlon.net/pages/viewpage.action?spaceKey=DATA&title=IAM+Security+Strategies), you need to authenticate to AWS and assume a role before using AWS resources (Sagemaker, S3...). You need to use `saml2aws` (check the link provided just above) to get your temporary token and assume a role (if you're using the `modeling` repository, we can assume you can get the `FCST-DATA-SCIENTIST` role) :
```sh
saml2aws login --force
```

* Assuming all your datasets from [Decathlon Demand Forecast - Forecast United Refining](https://github.com/dktunited/forecast-data-refining-demand/), you can execute the training & inference
```sh
python main.py --environment {env} --list_cutoff {list_cutoff}
```
> Notes : 
> * `env` will match one of the YAML configuration files in `config/`, so it can be `dev`, `prod`...
> * `list_cutoff` must be :
>   * a list of cutoff in format YYYYWW (ISO Format) between brackets AND simple quotes and **without spaces**, e.g. '[201925,202049,202051]'
>   * or the string `today` (it will match the current week)


## Development

* Clone the repository
```sh
git clone https://github.com/dktunited/forecast-modeling-demand/
cd forecast-modeling-demand/
```

* Switch to the `develop` branch or create a new branch from master
```sh
git checkout develop
git checkout -b myNewBranch master
```

The whole project works from a `newFeatureBranch` > `release` > `master` logic with mandatory reviewers for pull requests to `release` and `master`.

## Architecture
Please refer to the [Architecture wiki page](https://github.com/dktunited/forecast-modeling-demand/wiki/Architecture).

## Contribute

Please check the [**Contributing Guidelines**](https://github.com/dktunited/forecast-modeling-demand/blob/master/.github/markdown/CONTRIBUTING.md) before contributing.

Thanks goes to these wonderful people :

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/benbouillet"><img src="https://avatars2.githubusercontent.com/u/15980664?s=460&u=b546f8d2dd933638bfb73a76a3e7849d98bc0745&v=4" width="100px;" alt=""/><br /><sub><b>Ben Bouillet</b></sub></a><br /><a href="https://github.com/dktunited/forecast-modeling-demand/commits?author=benbouillet" title="Code">💻</a>
    <td align="center"><a href="https://github.com/Antoine-Schwartz"><img src="https://avatars0.githubusercontent.com/u/47638933?s=460&v=4" width="100px;" alt=""/><br /><sub><b>Antoine Schwartz</b></sub></a><br /><a href="https://github.com/dktunited/forecast-modeling-demand/commits?author=Antoine-Schwartz" title="Code">💻</a>
    <td align="center"><a href="https://github.com/BenjaminDeffarges"><img src="https://avatars1.githubusercontent.com/u/55504017?s=460&v=4" width="100px;" alt=""/><br /><sub><b>Benjamin Deffarges</b></sub></a><br /><a href="https://github.com/dktunited/forecast-modeling-demand/commits?author=BenjaminDeffarges" title="Code">💻</a>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

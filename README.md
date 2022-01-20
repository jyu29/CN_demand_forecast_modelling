## Readme 修订

因为POC阶段 modeling的运行不是透过jenkkins, 所以how to run的部分我没有预想架构。
每一个区块我会用中文说明一下，大部分都是帖log就行。
你本来的readme 内容我放在最底下没有删掉，如果你需要的话。



# Forecast Data Exposition CN

## Context
* [1. Description](#1-description)
* [2. Input & Output Data](#2-input-and-output-data)
* [3. Code Architecture](#3-code-architecture)
* [4. How to run](#4-how-to-run)
    * [4.1. xxxxxx](#41-xxxxxx)
    * [4.2. xxxxxxxxx](#42-xxxxxxxxx)
* [5. Common Error](#5-common-error)
* [6. What has been changed from master branch](#6-what-has-been-changed-from-master-branch)
<br>

## 1. Description

.........

Besides, if you want to know more about this pipeline's opreation, you can check these info:<br>
(any picture or other info link)
<br>


## 2. Input and Output Data

<table>
  <thead>
    <tr>
        <th>Type</th>
        <th>TableName</th>
        <th>Columns</th>
        <th>S3_Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=2>Input data</td>
      <td>predict-{timestamp}.josn</td>
      <td>model_id <br> start <br> target <br> cat <br> dynamic_feat</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/quicktest/quicktest-deepar-202038/input/</td>
    </tr>
    <tr>
      <td>.....</td>
      <td>.....</td>
      <td>.....</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td rowspan=4>Output_data</td>
      <td>forecast_deepar</td>
      <td rowspan=3>.....</td>
      <td rowspan=5>.....</td>
    </tr>
    <tr>
      <td>forecast_hw</td>
    </tr>
    <tr>
      <td>forecast_deepar-hw</td>
    </tr>
    <tr>
      <td>restruct_sales</td>
      <td>.....</td>
    </tr>    
  </tfoot>
</table>
</br>

## 3. Code Architecture 

```
forecast-data-exposition-quicktest
│    .gitignore
│    Jenkinsfile
│    Jenkinsfile_only_sql
│    spark_submit_create_table.sh
|    main.py 
│    enviroment.yml 
│    requirements.txt
│    README.md
│
├───config
│       dev.yml
│       prod.yml
|       whitelist.yml
│
└───src
       exposition_hankdler.py
       create_table.py
       utils.py
       
```

## 4. How to run


### 4.1. xxxxxx

   (如果要事先开启什么EMR或者Sagemaker可以写这, 像是开启EMR需要特定参数，然后会显示特定log)
 
   
   ```
   parameters: [
                    string(name: "nameOfCluster", value: "${BUILD_TAG}"),
                    string(name: "versionEMR", value: "emr-6.4.0"),
                    string(name: "ClusterType", value: "batch_cluster"),
                    string(name: "instanceTypeMaster", value: "c6g.8xlarge"),
                    string(name: "masterNodeDiskSize", value: "128"),
                    string(name: "nbrCoreOnDemand", value: "6"),
                    string(name: "nbrCoreSpot", value: "0"),
                    string(name: "instanceTypeCore", value: "r6g.8xlarge"),
                    string(name: "coreNodeDiskSize", value: "128"),
                    string(name: "nbrTaskNode", value: "0"),
                    string(name: "instanceTypeTask", value: "r5.2xlarge"),
                    string(name: "taskNodeDiskSize", value: "64"),
                    string(name: "hdfsReplicationFactor", value: "3")
                    ]
   ```

   
   > check pipeline console to see the log. 
   
   
   ```
      Your EMR dev-cluster-emr is built !
      Your available URI are :
      http://IBENKH18-ganglia.forecast-emr.subsidia.org/ganglia/
      http://IBENKH18-hue.forecast-emr.subsidia.org
      http://IBENKH18-node.forecast-emr.subsidia.org
      http://IBENKH18-spark.forecast-emr.subsidia.org
      http://IBENKH18-hdfs.forecast-emr.subsidia.org
      L'ip de votre cluster est : 10.226.xxx.xxx'
      By !
      Jenkins
      [Pipeline] }
      [Pipeline] // withAWS
      [Pipeline] sh
      + echo CLUSTER_IP=10.226.xxx.xxx
      CLUSTER_IP=10.226.xxx.xxx
   ```
      
    

### 4.2. xxxxxxxxx

   (实际执行步骤，像是jenkins的参数以及执行前的config档案修订，更新白名单之类的。)
    
   > Confim parameters in config file and push it. choose the file depend on your environment.(for example, `dev.yml` )
   
   > Confirm your bucket, path and name are all right.
   
   ```
   s3_path_parameters:
   buckets:
   refined: fcst-workspace
   exposition: fcst-workspace
   paths:
   refined_global:
   forecast-cn/fcst-refined-demand-forecast-dev/global/
   refined_specific: forecast-cn/fcst-refined-demand-forecast-dev/specific/
   file_setting:
   load_name:
   - predict-2021-12-14-12-15-12-419
   - predict-2021-12-14-08-38-28-448
   save_tag:v112
   ```
   
   
   > Remember to **Push and commit** the code from your IDE to the github branch name your entered .Then click the bottom **"bulid"** on the Jenkins 

      
   > (查看运行结果， log放来就行)


 **exposition handler**
 
 ```
         ####################################################################################
         ########## RECONSTRUCTED SALES EXPOSITION FOR CUTOFF 202146 AND ALGO deepar ##########
         ####################################################################################
         Formatting reconstructed sales dataframe...
         Building outputs for channel 'sac'...
           Mapping forecast dataframe...
             Filtered reconstructed sales horizon : '201947'
         Writing outputs for channel 'sac...'
         ###########################################################################################
         ############################## DEMAND FORECAST GLOBAL EXPOSITION ###########################
         #############################################################################################
         Building outputs for channel 'sac'...
         Writing outputs for channel 'sac...'
   ```
      

## 5. Common error

（这个就也是贴上log，提个解决方式一下就行。）

**error 1:**

   ```
   # first error message
   Traceback (most recent call last):
      ......
      KeyError: 'filename'
   Traceback (most recent call last):
      ......
   ValueError: Wrong number of items passed 2, placement implies 1
   
   # second error message   
   file name: predict-2021-12-01-12-42-18-742 not in folder, use next file name to load.
   file name: predict-2021-12-01-13-48-08-809 not in folder, use next file name to load.
   Traceback (most recent call last):
      .....
   OSError: Passed non-file path: fcst-workspace/None
   ValueError: No objects to concatenate
   
   # third errro meassage  
   Formatting forecast dataframe...
   Traceback (most recent call last):
      .......
   ValueError: Length mismatch: Expected 6454 rows, received array of length 43920
   ```
   <br>

> `Exposition_handler.py` can't find your data source path, or it got not entire path.debug file path:`{HTSAI/Sagemaker/exposition/expo_debug.ipynb}`.check if you got both path of input file and output file when you start the `exposition_handler`.If not entire input or output path, you may choose error week_id when you build the pipeline on jenkins or choose error model output name in your config file.spark config start lag, or it can't get enough resource to run the pipeline.




## 6. What has been changed from master branch

（改动的部分，主要都是备注在程式里，我这里只是清单列一下）

#### add parameter to choose input data source file name
  - `Exposition_handler.py` : add load_name and save_tag to decide which model's output you want to be the table source.
  - `bi_create_table.py` : add load_path and save_path to decide BI table's path in s3.  <br>

#### modify the BI table
  - due to chinese apo data from s3 can't be use, join the apo data from local-site. 
  - delete the redundant result table `outlier_model` and `cutoff_numweek_fcststep` 
  - add a new tmp table `realized_sales` to be a table store real_quantity.
  - group by the  and model_id to sum the column `real_quantity` of online and offline in table `model_week_sales`. <br>

#### add filter to reduce the size of BI table 
  - add `whitelist` and `blacklist` to filter what model_id you want in BI table.
  - filter forecast value by `deepar` and `apo_gd` in BI table `quantity_forecast_sales` and `f_forecast_global_demand`. 
  - filter realized value by `y` and `y_1` in BI table `quantity_forecast_sales`. 
  - only keep `<= 10` `forecast_step` value in BI table. <br>
 
#### make the process of creating BI table work automatically 
  - rewrite the `bigquery sql script` to the pysparkSQL script. 
  - compose the task of creating BI table and original jenkins pipeline <br>










<br><br>
-------------------------------
<br><br>



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


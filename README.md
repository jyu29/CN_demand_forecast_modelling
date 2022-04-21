# Decathlon Demand Forecast CN

## Context
* [0. Project slide](#0-Project Slide)
* [1. Description](#1-description)
* [2. Methdology Introduction](#2-methdology-introduction)
    * [2.1. Model framework](#21-model-framework)
    * [2.2. Cold start reconstruction](#22-cold-start-reconstruction)
    * [2.3. Lockdown reconstruction](#23-lockdown-reconstruction)
* [3. China adaption](#3-china-adaption)
* [4. Code Architecture](#4-code-architecture)
* [5. Input & Output Data](#5-input-and-output-data)
    * [5.1. Input Data](#51-input-data)
    * [5.2. Input File](#52-input-file)
* [6. How to trigger and make adaption](#6-how-to-trigger-and-make-adaption)
    * [6.1. Task trigger](#61-task-trigger)
    * [6.2. Make adaption](#62-make-adaption)
<br>

## 0. Project Slide

Links: [**Demand Forecast 2022**](https://docs.google.com/presentation/d/1ltZ2-cWDkzBmHQG7NZTweTnJ1tmen5OtiwcYDzwVNCY/edit#slide=id.gd1c50f97a9_2_3857) 


## 1. Description

A Sagemaker-backed, DeepAR-based, machine learning application to forecast sales for the Decathlon Demand Team.



Model used - [**DeepAR**](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html) developed by **AWS**.

This branch is to manage the development of China adaption.
<br>

## 2. Methdology Introduction
### 2.1 Model framework
In model-wise, DeepAR is used to inference weekly sales in 16 weeks, and holt-winters is used to inference in 104 weeks. In result-wise, forecast generated by DeepAR is directly used in 10 weeks, and forecast generated by holt-winters is used for 16 to 104 weeks. **A stacking model is built to generate the forecast between 11 and 15 weeks by leverage both the output of DeepAR and holt-winters:**

![](http://latex.codecogs.com/svg.latex?weight_i=\frac{W_i-W_0}{N})

![](http://latex.codecogs.com/svg.latex?\hat{y_i}=\hat{y_{si}}*(1-weight_i)+\hat{y_{li}}*weight_i)

- ![](http://latex.codecogs.com/svg.latex?W_0) : first stacking forecast step (for now is 11)
- N: number of stacking weeks
- ![](http://latex.codecogs.com/svg.latex?\hat{y_{si}}) : forecast generated by DeepAR in Week i
- ![](http://latex.codecogs.com/svg.latex?\hat{y_{li}}) : forecast generated by Holt-Winters in Week i

### 2.2 Cold start reconstruction
The objectives of cold start sales reconstruction:
- Replace the sales in the first 8 weeks after new product launch, to emit the effect of implementation period
- Make sure that there are at least 156 weeks of selling histories

To reconstruct the sales of new model, firstly it will calculate the average weekly sales of the same family, and then will calculate the new models' sales share of the average family sales. The share will act as an index to scale up and down the sales curve of the average family sales:
- Calculate the weekly average model sales within each family, then we will get a **cluster sales quantity**
- Since there are weeks of sales of the new products (at least 1 week), to compare the sales between cluster sales quantity and sales of new products each week, then we will get a scale factor each week (sales of new products / cluster sales quantity), which means we will get a series of scale factor by the end of this step
- Average all the scale factors to generate a final scale factor for each model
- Then the reconstructed sales in each week will be: **scale factor** * **cluster sales quantity**


### 2.3 Lockdown reconstruction
Currently use a LGBM model to predict sales within the lock-down periods. The training data is the sales in the last year of lockdown.

Feature Name | Data Type | Description
-- |-- | --
model_id | int | the id of the model
family_id | int | the id of the subdivided category the model belongs to
sub_department_id | int | the id of the subdepartment the model belong to
department_id | int | the id of the department the model belong to
univers_id | int | the id of the universe the model belong to, which stands for the subdivided sport
product_nature_id | int | The id of the product nature of the model, which shows the use
time_idx | int64 |The number of weeks from the earliest date in the database
age | int64 | the number of weeks the model has been sold since the first week
week | int64 | the week of the year 
month | int64 | the month of the year
min_sales_quantity | int64 | the minimum weekly sales quantity of the model in all selling weeks
max_sales_quantity | int64 | the maximum weekly sales quantity of the model in all selling weeks
mean_sales_quantity | float64 | the average weekly sales quantity of the model in all selling weeks
median_sales_quantity | float64 | the median of weekly sales quantity of the model in all selling weeks
std_sales_quantity | float64 | the standard deviation of weekly sales quantity of the model in all selling weeks
sales_quantity_lag | float64 | the sales quantity of the model in the same week in last year
avg_qty_lag_by_family_id | float64 | the average of the sales_quantity_lag  of all models having the same family_id as the model
avg_qty_lag_by_sub_department_id | float64 | the average of sales_quantity_lag of all models having the same sub_department_id as the model
avg_qty_lag_by_department_id | float64 | the average of sales_quantity_lag of all models having the same department_id as the model
avg_qty_lag_by_univers_id | float64 | the average of sales_quantity_lag of all models having the same univers_id as the model
avg_qty_lag_by_product_nature_id | float64 | the average of sales_quantity_lag of all models having the same product_nature_id as the model


## 3. China Adaption

1) **Channel split**

The offline and online sales are split since the global refining part, thus in table model_week_sales we will aggregrate sales by offline and online seperately. We will train and inference by channel, and then we aggreate the forecast of both offline and online together as the final forecast. 

2) **Add global dynamic features - num_holiday_weekend, and num_holiday_weekday**

To mitigate calendar gap year on year, another 2 global dynamic features are added:
- num_holiday_weekend: holiday count within weekend, the iead is to identify the holiday trade-off
- num_holiday_weekday: holiday count within weekdays
For more details about calculating these two dynamic features, you can refer the code at ./optimization/optimization_notebook/01_calendar_gap.ipynb

3) **Add static features - seasonality label**

The idea is to club all the products into several groups based on their **sales shape in the previous year**. To make sure there is a complete time series in the previous years for all the products, the segmentation is generated after the cold-start sales reconstruction.

The segmentaion is built with a K-means model (n_cluster=5), leveraging features:

Feature Name | Data Type | Description
-- | -- | --
model_id | int64 | the id of the model
max_sales_week | int64 | the week with the maximum weekly sales of the model in the year
min_sales_week | int64 | the week with the minimum weekly sales of the model in the year
num_peek_sales | int64 | the number of weeks in the year when the weekly sales of the model is larger than 80% of its maximum weekly sales of the year
num_low_sales | int64 | the number of weeks in the year when the weekly sales of the model is lower than 120% of its minimum weekly sales of the year
std_sales_g | int64 | if the standard deviation of weekly sales quantity of the model in all selling weeks is less than 10, it equals to 1, else 0

For more details about calculating these two dynamic features, you can refer the code at ./optimization/optimization_notebook/02_time_series_segmentation.ipynb


## 4. Code Architecture 

```
forecast-data-exposition-quicktest
├──Jenkinsfile
    ├──main.py
    ├──enviroment.yml 
    ├──README.md
    ├──pytest.ini
    ├──main_seed.ipynb    
├──config
    │- dev.yml
    │- prod.yml
    |- seed.yml    
├──deployment
    ├──sagemaker_arima
        ├──build_image.sh
        ├──Dockerfile
        ├──requirements.txt
    ├──sagemaker_hw
        ├──build_image.sh
        ├──Dockerfile
        ├──requirements.txt   
├──src
    ├──data_handle.py
    ├──main_sagemaker_arima.py
    ├──main_sagemaker_hw.py
    ├──outputs_stacking.py
    ├──refining_specific_functions.py
    ├──sagemaker_arima_functions.py
    ├──sagemaker_hw_functions.py
    ├──sagemaker_utils.py
    ├──utils.py
├──test
    ├──data
    ├──test_data_handle.py
    ├──test_outputs_stacking.py
    ├──test_refining_specific_functions.py
    ├──test_sagemaker_utils.py
    ├──test_utils.py 
├──optimization
    ├──optimization_notebook
        ├──01_calendar_gap.ipynb
        ├──cn_holiday.yaml
        ├──02_time_series_segmentation.ipynb
    ├──performance_analysis
        ├──performance_analysis.ipynb
        ├──codeA_2021.yaml     
```
Important scripts to know:
- `main.py` is the main trigger, which will fetch functions from `src.data_handler.py`, `src.sagemaker_utils.py` and `src.outputs_stacking.py`
    - `src.data_handler.py` is the main scripts about modeling data refining, and it fetchs function from `src.refining_specific_functions.py`
    - `src.sagemaker_utils.py` stores functions about sagemaker training and inference job setting
    - `src.outputs_stacking.py` is a scipt about stacking short-term and long-term result to generate the mid-term result
- `src.utils.py` is a script stores all common functions, i.e. read and write data from S3, datetime datatype transformer


## 5. Input and Output Data

### 5.1 Input data
#### DeepAR
#### 5.1.1 Input data format of DeepAR
In DeepAR, input data should be stored in json lines format and contain the following fields:
- `start` - a string with format YYYY-MM-DD HH:MM:SS
- `target` - an array of  floating-point values or integers that present the time series
- `dynamic_feat`(optional) - an array of arrays of floating-point values or integers that represents the vector of custom feature time series (dynamic features). There are 2 kinds of dynamic features: 
    - global dynamic feature: features in time granuality, which means all the tiem series(models) share the same global dynamic features
    - specific dynamic feature: features in time and model level, which means each model will have its specific dynamic features along the time series
- `cat`(optional) - an array of categorical features that can be used to encode the groups that the records belong to. Categorical features must be encoded as a 0-based sequence of positive integers.


#### 5.1.2 Input data sets
Currently, we mainly leverage 5 datasets to generate the time series, static and dynamic features:

#|input date| columns |description
-- |-- | -- | --
1|model_week_sales | model_id, week_id, date, channel, sales_quantity |get sales history to generate the time series
2|model_week_tree | model_id, week_id, family_id, sub_department_id, department_id, univers_id, product_nature_id, model_label,family_label, sub_department_label, 3|department_label,univers_label, product_nature_label, brand_label, brand_type| to the product tree information of models
4|model_week_mrp | model_id, week_id, is_mrp_active | get the MRP status of models to filter out active models
5|offline_reconstructed_sales_lockdowns | model_id, week_id, date, sales_quantity_reconstructed | reconstructed offline sales in covid shutdown period (2020W5-2020W17)
6|online_reconstructed_sales_lockdowns | model_id, week_id, date, sales_quantity_reconstructed | reconstructed online sales in covid shutdown period (2020W5-2020W17)
7|df_segmentation | model_id, seasonality label | adding static features - seasonality label


#### 5.1.3 Categorical features and dynamic features
`Categorical features`:
- family_id
- sub_department_id
- department_id
- univer_id
- product_nature_id
- seasonality label: segmentaion label, int ranging from 0 to 4

`Global dynamic features`:
- num_holiday_weekend: holiday count within weekend, the iead is to identify the holiday trade-off
- num_holiday_weekday: holiday count within weekdays

`Specific dynamic features`:
- is_rec: whethter the sales is reconsturcted

### 5.2 Input file
<table>
  <thead>
    <tr>
        <th>Algorithm</th>
        <th>Type</th>
        <th>TableName</th>
        <th>Columns</th>
        <th>S3_Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td rowspan=6>DeepAR</td> 
    </tr>        
    <tr>
      <td rowspan=3>Input data</td>
      <td>train-{timestamp}.josn</td>
      <td>model_id <br> start <br> target <br> cat <br> dynamic_feat</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-deepar-{cut_off}/input/</td>
    </tr>
    <tr>
      <td>predict-{timestamp}.josn</td>
      <td>model_id <br> start <br> target <br> cat <br> dynamic_feat</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-deepar-{cut_off}/input/</td>
    </tr>
    <tr>
      <td>times_segmentation.parquet</td>
      <td>model_id <br> seasonality_label</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-deepar-{cut_off}/input/</td>
    </tr>
      <td rowspan=2>Output data</td>
      <td>model.tar.gz</td>
      <td>/</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-deepar-{cut_off}/model/{run_name}-deepar-{cut_off}-{timestamp}/</td>
    <tr>
      <td>predict-{timestamp}.josn.out</td>
      <td>mean <br> 0.1 <br> ... <br> 0.8 <br> 0.9</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-deepar-{cut_off}/output/</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
        <td rowspan=6>Holt-Winters</td> 
    </tr>        
    <tr>
      <td>Input data</td>
      <td>train-{timestamp}.parquet</td>
      <td>model_id <br> start <br> target <br> cat <br> dynamic_feat</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-hw-{cut_off}/input/</td>
    </tr>
      <td rowspan=2>Output data</td>
      <td>model file</td>
      <td>/</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-hw-{cut_off}/model/{run_name}-hw-{cut_off}-{timestamp}/</td>
    <tr>
      <td>predict-{timestamp}.parquet.out</td>
      <td>model_id <br> forecast_step <br> forecast</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-hw-{cut_off}/output/</td>
  </tbody>
  <tbody>     
    <tr>
      <td>Stacking - final output</td> 
      <td>Output data</td>
      <td>predict-{timestamp}.parquet.out</td>
      <td>model_id <br> forecast_step <br> forecast</td>
      <td>s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/specific/{run_name}/{run_name}-deepar-hw-{cut_off}/output/</td>
    </tr>
  </tbody>
</table>
</br>


## 6. How to trigger and make adaption
### 6.1. Task trigger

#### 6.1.1 To trigger with shell

`main.py` is the script linking all other required scripts. Therefore, to trigger the modeling pipeline is to trigger the `main.py`. Before running the modeling part in Sagemaker, you are required to build a virtual environment via conda at first:

```sh
# Set-up env conda SM
envName=pytorch_forecasting
conda env create -f environment.yml --prefix ~/SageMaker/kernels/$envName
ln -s ~/SageMaker/kernels/$envName ~/anaconda3/envs/$envName
python -m ipykernel install --user --name $envName --display-name $envName
# Then restart the notebook instance (with lifecycle set to 'persistent-conda-kernels')
# ==> The kernel is available as $envName
```
You can find the `environment.yml` under the modeling repo. Please ensure you have go to the same folder with `environment.yml` to run the cmd above.


Now, you are able to trigger the job:
`Example:`
```sh
# 1. Activate virtual environment
conda activate /home/ec2-user/SageMaker/kernels/pytorch_forecasting

# 2. Go to the sub-folder of main.py
cd SageMaker/modeling/forecast-modeling-demand

# 3. Trigger the job
python main.py 'dev' [202111,202112,202113,202114,202115,202116,202117,202118,202119,202120] 'piloted' 'online'
```
To trigger `main.py`, 4 parameters are required:

config | definition
-- | --
'dev' | config_name, must be in 'dev','prod' or 'seed'
[202111,...,202120] | list of cut_offs
'piloted' | run_name
'online' | channel, must be 'online' or 'offline'


#### 6.1.2 Jenkins trigger (To be updated)


### 6.2 Make adaption

There are several ways to add new features and we are not going to go through all the stuffs. The idea is to provide some examples could be leveraged.

#### 6.2.1 Adding static features
If you would like to add a static feature, the impacted script is `data_handler.py` (impacted function: refining_specific), adding the additional dataframe into the dictionary ***static_features***. 
`Example:`
```sh
# impacted function: refining_specific
if self.static_features:

    ### Add static features
    self.static_features['seasonality_label'] = df_segmentation[['model_id','seasonality_label']]
    ###
    
    df_static_features = df_target[['model_id']].drop_duplicates().copy()
    for feature_name, df_new_feat in self.static_features.items():
        df_static_features = self._add_feature(df_features=df_static_features,
                                               df_new_feat=df_new_feat,
                                               feature_name=feature_name)
```                                                       
For the seasonality label case:
- Add a new function ***time_series_segmentation*** into script `refining_specific_functions.py` that used to generate the new static feature dataframe
- Import this function into script `data_handler.py`, and update function ***refining_specific*** by adding:
```sh
# Generate other static features
df_segmentation = time_series_segmentation(df_sales, self.cutoff)
```

#### 6.2.2 Adding global or sepcific dynamic features

If you would like to add a global dynamic feature, and the new feature is pre-calculated. We can just update the ***global_dynamic_features*** in `main.py`:
- At first, put the additional data source to S3 bucket (suggested path: s3://fcst-workspace/forecast-cn/fcst-refined-demand-forecast-dev/global)
- Update `main.py`:
```sh
CALENDAR_GAP_PATH = f"{REFINED_DATA_GLOBAL_PATH}calendar_gap.parquet"
df_calendar_gap = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, CALENDAR_GAP_PATH)

# /!\When adding add features to the dictionary, please ensure the column name of the features in the dataframe is the same as the key of the feature in the dictionary
global_dynamic_features = {
    'num_holiday_weekend':{'dataset':df_calendar_gap[['week_id','num_holiday_weekend']],'projection':'as_provided'},
    'num_holiday_weekday':{'dataset':df_calendar_gap[['week_id','num_holiday_weekday']],'projection':'as_provided'}
}
```

If the new feature is required to calculated in the piepline, the pipeline integration methdology is the same as the previous session ***Adding static features***.

The methdology of adding a specific dynamic features, the way is almost the same as adding a global dynamic feature. The only difference is to generate the dynamic feature dataframe. For global dynamic features, the required schema is model_id | feature_name, while for specific dynamic features, the required schema is model_id | week_id | feature_name.

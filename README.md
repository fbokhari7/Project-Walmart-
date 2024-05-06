This project aims to analyze the sales data of one of the largest retailers globally, focusing on understanding the factors influencing revenue. We will investigate whether variables such as air temperature, fuel cost, consumer price index (CPI), seasonal discounts, and the presence of holidays impact the sales performance of this major retail chain. By leveraging machine learning, we aim to identify key factors that contribute to revenue generation and explore how these insights can be used to minimize costs and maximize economic impact.

The dataset includes information from 45 Walmart stores, including weekly sales figures, air temperature, fuel prices, CPI, and unemployment rates in their respective regions. Our analysis will provide valuable insights into the retail industry's dynamics, offering strategies to enhance business performance and optimize decision-making processes.

### Data Dictionary:
The data source for this project is the Walmart Sales dataset available on Kaggle. The dataset contains information from 45 Walmart stores, including weekly sales figures, air temperature, fuel prices, CPI, and unemployment rates in their respective regions. The dataset can be accessed at the following link: 
https://www.kaggle.com/datasets/mikhail1681/walmart-sales/data

The dataset used for model building contained 6435 observations of 8 variables. The data contains the following information:

| **Features** | **Data Description** | 
|:--------:|:--------:|
|  Store  |  This column contains the store number.   |  
|  Date   | This column contains the sales week start date.   |  
|  Weekly_Sales   |  This column contains the weekly sales figures.   |  
|  Holiday_Flag   |  This column indicates the presence or absence of a holiday. In this 1 indicates a holiday and 0 indicates no holiday.   |  
|  Temperature   |  This column contains the air temperature in the region where the store is located.  |  
|  Fuel_Price   |  This column contains the fuel cost in the region where the store is located.    | 
|  CPI  |  This column contains the consumer price index for the region. |  
|  Unemployment  | This column contains the unemployment rate for the region.  |  


Repository Contents: data: Folder containing the dataset. scripts: Python scripts for data preprocessing, data analysis, and evaluation. 
Pdf: A pdf of the whole script and visualizations.

List of Python packages required to run the code. 

Pandas: Library for data manipulation and analysis. 

Matplotlib: Library for creating visualizations in Python. 

Seaborn: Data visualization library for drawing statistical graphics.

Scikit-learn: For machine learning tasks.

Contributions:
Feel free to contribute by adding new features or exploring different algorithms to enhance the analysis of Walmart's sales data.

License:
This project is licensed under the MIT License.

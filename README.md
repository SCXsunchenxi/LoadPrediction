# LoadPrediction

## Data

We disclosed some training and test data in the project. They are in the 'data' folder.  
In our project, we used the flight data from 5 aircraft (P123, P124, P125, P126 and P127), each aircraft having 20 flights in different days. But due to confidentiality, we only provide flight data from 2 aircraft (P125 and P127), each aircraft having 2 flights in different days. Each file contians flight parameters and strains in one flight of an aircraft. Besides, we change the chronological order of the provied data. More information is listed in 'DataInfo.doc'.   

## Algorithm

We provided our code files. They are in the 'algorithms' folder.  
1. Data process: 'data_process.py' processes raw data by data cleaning, data integration, data augmentation, data normalization and data division.  
2. Prediction method: 'create_model.py' builds multi-model to predict strains from flight parameters. It calls 'MultilayerPerceptron.py', 'LightGBM.py' and 'RidgeRegression.py' through 'base_model.py'.  
3. Calibration method: 'indirectmodel_calibration.py' calibrates the strain coefficients by 'make_strain_pair.py', 'find_k.py' and 'DT_explanation.py'.  

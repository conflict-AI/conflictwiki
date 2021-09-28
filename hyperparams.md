# Model Hyperparameters

|   | Dyadic Model     | Systemic Model | Ceiling Model |
| :----:  |  :----:   |   :----:   |   :----: |
|  node / endge encoder |   500 &rightarrow; 128 &rightarrow; 64    |   500 &rightarrow; 128 &rightarrow; 64   | 500 &rightarrow; 128 &rightarrow; 64 |
|  edge classifier |  64 &rightarrow; 64   |    64 &rightarrow; 6     | 64 &rightarrow; 6 |
| learning rate | 0.001  | 0.001 | 0.001 |
| message passing steps |- | 2  |2|
| batch size  | 512| 512 | 512 |
| epochs | 30 | 30 | 30 |
| early stopping patience |3| 3| 3|

The grid search involved sweeping over the following hyperparameter space:

|   | Options |
| :----:  |  :----:   | 
|  node / endge encoder last layer |  32, 64, 128, 256, 512 |
|  edge classifier last layer  |  6, 12, 24, 48, 96 |
| learning rate | 0.01, 0.001, 0.0001 | 
| message passing steps | 1,2,3|


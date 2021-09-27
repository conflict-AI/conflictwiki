
|   | Dyadic Model     | Systemic Model | Ceiling Model |
| :----:  |  :----:   |   :----:   |   :----: |
|  node / endge encoder |   500 &rightarrow; 128 &rightarrow; 64    |   500 &rightarrow; 128 &rightarrow; 64   | 500 &rightarrow; 128 &rightarrow; 64 |
|  edge classifier |  64 &rightarrow; 64   |    64 &rightarrow; 6     | 64 &rightarrow; 6 |
| learning rate | 0.001  | 0.001 | 0.001 |
| message passing steps |- | 2  |2|
| batch size  | 512| 512 | 512 |
| epochs | 30 | 30 | 30 |
| early stopping patience |3| 3| 3|


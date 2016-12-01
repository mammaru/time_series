# pyts
A object oriented library to analyse multivariate time series data.

* EM algorithm
* Generalized cross-validation
* Vector auto regressive model
* State space model(Dynamic linear model)
  * Kalman's algorithm(prediction, filtering and smoothing)
  * Particle Filter

# Usages

```
from pyts.ssm import SSMKalman as SSM
ssm = SSM(observation-dimention, system-dimention)
em = EM()
em(ssm, data)
```



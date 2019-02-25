# CIFAR-10
Small Image Detection Models Using Deep Learning
<p>[Data Source:](https://www.cs.toronto.edu/~kriz/cifar.html)

## Jupyter Notebooks
- [Covnet Using Data Augmentation](cifar10_COVNET_no_noise.ipynb) (Accuracy: 92.16%)
- [ResNetv1(20)](CIFAR_10_RESNETv1b_No_Noise.ipynb) (Accuracy: 91.83%)
- [ResNetv2(29)](CIFAR_10_RESNETv2c_No_Noise.ipynb) (Accuracy: 93.26%)

## Model Files
- Architecture
- Weights

<p> To be read as follows:
  
  
```python
json_string = pickle.load(open('covnet_model_json.pkl',"rb"))
model = keras.models.model_from_json(json_string)

model.compile(
   loss='categorical_crossentropy', 
   optimizer=keras.optimizers.Adam(), 
   metrics=['accuracy']
)
```
<p>TBD

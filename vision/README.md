# Deeplearning Vision
* Data directory: `data/<problemID>/`
* Model directory: `nnmodels/<problemID>/datasetID/`


## Traffic Sign Classification

**Directory Structure**
* `problemID:traffic-signs`
* `datasetID:BelgiumTSC`
	- `data/traffic-signs/BelgiumTSC/Training`
	- `data/traffic-signs/BelgiumTSC/Testing`
* **Architecture: simple**
	* datasetID: BelgiumTSC
  * accuracy: 0.40 to 0.70
  * framework: TensorFlow v1.9
* References:
  * https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6
  * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
* **Run:**
1. Download and Extract the data:
```bash
source get.traffic-signs.BelgiumTSC.sh
```
2. Train and Predict
```bash
python traffic-signs-tf-simple-v2.py
```
  - Change `FLAGS['n_epochs']` in the 'traffic-signs-tf-simple-v2.py' python file with higher values. Default : `FLAGS['n_epochs'] = 21`
  
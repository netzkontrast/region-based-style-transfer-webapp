
## Main Algorithms:

- Semantic Segmantation
- Style Transfer
- Color Transfer
- Image blending (feathering)

## Usage:

Clone this repo (be patient, this repository is about 4-5GB using git-lfs to store large checkpoint files):

```
   git clone git@github.com:zhichengMLE/region-based-style-transfer-webapp.git
```

Frontend: 
```
   cd ./region-based-style-transfer-webapp/frontend/
   HOST='0.0.0.0' PORT=8080 npm start
```

Backend:

```
   python main.py
```

NOTE:

> You need to change the url of frontend to receive from backend. (public url of GCP)


Live demo: http://35.245.64.0:8080/#/ (please check it out before Dec 15th, 2018)

<hr>
<hr>
<hr>

</br>

## 1. Problem Definition and Motivation
![1](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/1.png)

## 2. System Architecture
![2](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/2.png)


## 3. Methods
### 3.1. Semantic Segmentation
![3.1.](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/3.1.png)

### 3.2. Style Transfer
![3.2.](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/3.2.png) 

### 3.3. Color Transfer
![3.3](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/3.3.png)

### 3.4. Blending
![3.4.](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/3.4.png)

## 4. Experiment
![4.1](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/4.1.png)
![4.2](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/4.2.png)
![4.3](https://raw.githubusercontent.com/zhichengMLE/region-based-style-transfer-webapp/master/report/images/4.3.png)


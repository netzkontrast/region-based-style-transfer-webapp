
## Main Algorithms:

- Semantic Segmantation (including boundary optimization)
- Style Transfer
- Color Transfer
- Image blending (feathering)

## Usage:

Clone this repo (be patient, this repository is about 4-5GB using git-lfs to store large checkpoint files):

```
   git clone git@github.com:zhichengMLE/region-based-style-transfer-webapp.git
```

Install dependencies (use the one match the python version that you are using, I had only tested on Python3.)

```
   # python 2
   sudo pip install -r requirements.txt
   
   # python 3
   sudo pip3 install -r requirements.txt
```

Frontend: 
```
   cd ./region-based-style-transfer-webapp/frontend/
   
   # if you want to run it in GCP and make it public.
   HOST='0.0.0.0' PORT=8080 npm start
   
   # if you want to run it in local machine.
   npm run dev
```

Backend:

```
   python main.py
```

NOTE:

> You need to change the url of frontend to receive from backend. (public url of GCP)
> 
> If you have trouble to reproduce the result, please contact me via zd2212@columbia.edu


**Live demo: http://35.245.64.0:8080/ (Please check it out before Dec 20, 2018. If you had trouble to access it, probably because we are out of credit. In this case, you could follow the usage session to reproduce the result locally. If you have trouble to reproduce the result, please contact me via zd2212@columbia.edu)**


Website Screenshot:

![Website](https://github.com/zhichengMLE/region-based-style-transfer-webapp/blob/master/report/images/website.png)

![Website](https://github.com/zhichengMLE/region-based-style-transfer-webapp/blob/master/report/images/website2.png)

<hr>
<hr>
<hr>

</br>

> All the report could be found in folder: /report/

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


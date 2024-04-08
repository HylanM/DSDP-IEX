# Titanic Dataset Analysis

## **Introduction**

Hello! This Streamlit application runs through Docker and shows the classifications and visualizations of Titanic passengers survival. This data was downloaded from Kaggle and the link can be found on the first page of the project. 

## Running the Project

**1. Requirements**

* Python 3.12
* Streamlit (`pip install streamlit`)
* Pandas (`pip install pandas`)
* Numpy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)
* Seaborn (`pip install seaborn`) - visualizations
* Scikit-learn (`pip install scikit-learn`)

**2. Running with Docker**

(Assuming you have Docker installed)

* Build the Docker image:

```docker build -t my_streamlit_app  .```

* Run the container:

```docker run -p 8501:8501 my_streamlit_app```

*Important note when launching the program it may launch as 0.0.0.:8501 and not work. If you change it the code listed before the application should launch. 

Open http://localhost:8501 in your web browser to access the Streamlit app.


## Conclusions

* First Class Passengers had a higher chance of survival.

* Women and children had a higher chance of survival.

* Survival chance decreased with age.

* The higher the fare that was paid, the better chance of survival. 
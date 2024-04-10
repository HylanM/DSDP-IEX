# Housing Dataset Analysis

## **Introduction**

Hello! This Streamlit application is for the Ames Housing dataset and runs through Docker. This dataset goes over house prices in Ames, Iowa and how different factors impact price. This data was downloaded from Kaggle and the link can be found on the first page of the project. 

## Running the Project

**1. Requirements**

* Python 3.12
* Streamlit (pip install streamlit)
* Pandas (pip install pandas)
* Numpy (pip install numpy)
* Matplotlib (pip install matplotlib)
* Seaborn (pip install seaborn) 
* Scikit-learn (pip install scikit-learn)

**2. Running with Docker**

(Assuming you have Docker installed)

* Build the Docker image:

```docker build -t my_streamlit_app  .```

* Run the container:

```docker run -p 8501:8501 my_streamlit_app```

*Important note when launching the program it may launch as 0.0.0.:8501 and not work. If you change it the code listed before the application should launch. 

Open http://localhost:8501 in your web browser to access the Streamlit app.

**3. What to Expect**
This project included two pages, the first one is called Filters and has ways for the data to be filter, to see the different impacts that different characteristics have on the price.

![filter](Pictures/filter.png)

In addition, the second page is called Visualization.

![visual1](Pictures/visual1.png)

A few of the visualizations that you can expect to find is the area of the house compared to cost, as seen above and the number of bedrooms in the house. These visuals help exmaine the individual impact that different characters have on the cost of a house

![embark](Pictures/embark.png)

## Conclusions

* Individuals that embarked from cherbourg France had the highest rate of survival.

* individuals that boarded at Southhamptons, UK had the lowest survival rate

* Younger people had a higher rate of survival 

* The more money that was spent on the ticket the more likely that person was to survive 
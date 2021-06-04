# AB Analytics

This is a flask application that achieves the following:
1. Provides interactive visualisation for the customer, transaction and promotion offer data for the Starbucks loyalty app
2. Segments customers into profile clusters using machine learning, and serves the model so that the user can enter their details and be placed in a customer group
3. Personalised offer recommendations (Now LIVE) - for each customer in the database, a recommender system is used generate the best offers for that individual that will make them most likely to make a purchase.

## Link to app: 
The application is now hosted [here](https://analytics-ab.herokuapp.com) using Heroku.

## Running the app locally

### Clone the repository

```
git clone https://github.com/matthewgela/udacity-starbucks-ui.git
```

### Prerequisites

To install the flask app, you need:
- python3
- python packages in the requirements.txt file
 
 Install the packages with
``` 
 pip install -r requirements.txt
```

To start the flask server, execute the following line in the root folder of the repository

``` 
 python run_local.py
```

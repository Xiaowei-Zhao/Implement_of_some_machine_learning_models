# Implement_of_some_machine_learning_models

In LinearModel1.py and LinearModel2.py

I analyze data using the linear regression techniques. The goal
of the problem is to predict the miles per gallon a car will get using six quantities (features) about that
car.

In addition to linear regression, I also implement the ridge regression and modify my code to learn a pth-order polynomial regression model for p = 1; 2; 3

In naiveBayes_KNN_logisticRegression.py

I implement the naive Bayes classifier as well as the kNN algorithm and logistic regression algorithm. The data consists of examples of spam and non-spam
emails, of which there are 4600 labeled examples.

In every experiment below, I randomly partition the data into 10 groups and run the algorithm 10 different
times so that each group is held out as a test set one time.

In the last part, I implement an algorithm called 'Newton's method' for logistic regression.

In K_means.py

I implement the K-means algorithm. Generate 500 observations from a mixture of three
Gaussians on R^2 with mixing weights ¦Ð = [0:2; 0:5; 0:3] with different means and covariance.

In GMM_EM.py

I implement the EM algorithm for the Gaussian mixture model, with the purpose
of using it in a Bayes classifier. The class conditional density will be the Gaussian mixture model (GMM). In these experiments, please
I initialize all covariance matrices to the empirical covariance of the data being modeled. Then randomly
initialize the means by sampling from a single multivariate Gaussian where the parameters are the mean
and covariance of the data being modeled. Initialize the mixing weights to be uniform. Finally, I implement the EM algorithm for the GMM
and conduct prediction on test dataset

In Matrix_factorization.py

I implement the MAP inference algorithm for the matrix completion problem 
I train the model on the larger training set for 100 iterations. For each user-movie pair in the test set, predict the rating using the relevant dot
product. In a table, I show in each row the final value of the training objective function next to the RMSE on the testing
set. Then I sort these rows according to decreasing value of the objective function.
In the last part, for the run with the highest objective value, I pick the movies ¡°Star Wars¡± ¡°My Fair Lady¡± and
¡°Goodfellas¡± and for each movie find the 10 closest movies according to Euclidean distance using
their respective locations vj.


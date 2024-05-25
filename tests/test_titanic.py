import unittest
from titanic import train_score_model
from sklearn.linear_model import LogisticRegression

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.X_train = [[0, 1], [2, 3]]
        self.X_test = [[1, 0], [4, 2]]
        self.y_train = [[0, 1]]
        self.y_test = [[1, 0]]

    def test_train_score_model(self):
        model, score = train_score_model(LogisticRegression(), self.X_train, self.X_test, self.y_train, self.y_test)
        # Add assertions to check model and score
        print(score)
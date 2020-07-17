import numpy as np

# note - to add these in common evaluetion metrics file


def mae_regression(actual_value, predicted_value):
	"""

	Measure of Absolute Error
	
	"""
    return abs(predicted_value - actual_value)


def rmsle(actual_value, predicted_value):
	"""
	Root Mean Squared Logarithmic Error
	"""
    return sum(abs(np.log(predicted_value) - np.log(actual_value)))


def r2(actuel_value, predicted_value):
	"""
	R-Square or Cofficient of Determination

	Mathes Background -

	Explains the percentage of Total Variation is described by variation in X
	"""
    y_mean = mean(actual_value)
    den = sum((predicted_value - y_mean)**2)
    num = sum((predicted_value - actual_value)**2)
    return (1 - num/den)
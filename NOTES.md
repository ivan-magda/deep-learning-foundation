# Neural Networks

#### [Perceptron](https://en.wikipedia.org/wiki/Perceptron)
Data, like test scores and grades, is fed into a network of interconnected nodes.
These individual nodes are called [perceptrons](https://en.wikipedia.org/wiki/Perceptron) or neurons, and they are the basic unit of a neural network. *Each one looks at input data and decides how to categorize that data*.

#### Weights
We might be wondering: "How does it know whether grades or test scores are more important in making this acceptance decision?"
Well, when we initialize a neural network, we don't know what information will be most important in making a decision.
It's up to the neural network to *learn for itself* which data is most important and adjust how it considers that data.


When input data comes into a perceptron, it gets multiplied by a weight value that is assigned to this particular input. For example, the perceptron above have two inputs, tests for test scores and grades, so it has two associated weights that can be adjusted individually. These weights start out as random values, and as the neural network learns more about what kind of input data leads to a student being accepted into a university, the network adjusts the weights based on any errors in categorization that the previous weights resulted in. This is called training the neural network.


A higher weight means the neural network considers that input more important than other inputs, and lower weight means that the data is considered less important. An extreme example would be if test scores had no affect at all on university acceptance; then the weight of the test score input data would be zero and it would have no affect on the output of the perceptron.

#### Gradient Descent
Weight update can be calculated as: `Δwi = ηδxi`
with the error term δ as: `δ = (y − y^) f′(h) = (y − y^)f′(∑wixi)`

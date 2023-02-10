import nn
from  backend import Dataset
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.
        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)
        self.batch_size = 1
        
    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x) -> float:
        """
        Calculates the score assigned by the perceptron to a data point x.
        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)


    def get_prediction(self, x) -> int:
        """
        Calculates the predicted class for a single data point `x`.
        Returns: 1 or -1
        """
        dot_product_result = nn.as_scalar(self.run(x))
        return 1 if dot_product_result >= 0 else -1

    def train(self, dataset: Dataset) -> None:
        """
        Train the perceptron until convergence.
        """

        while True:
            learning = False
            for point_coordinate, constant_label in dataset.iterate_once(self.batch_size):
                label_prediction = self.get_prediction(point_coordinate)
                label =  nn.as_scalar(constant_label)

                if label_prediction != label:
                    self.w.update(point_coordinate, label)
                    learning = True

            if not learning:
                break
        
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        self.input_size = 1
        self.output_size = 1

        self.hidden_layer_size = 512
        self.batch_size = 200
        self.learning_rate = 0.05

        self.W1 = nn.Parameter(self.input_size, self.hidden_layer_size)
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.W2 = nn.Parameter(self.hidden_layer_size, self.output_size)
        self.b2 = nn.Parameter(1, self.output_size)

    def run(self, x) -> nn.AddBias:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        dot_product_1 = nn.Linear(x, self.W1)
        add_bais = nn.AddBias(dot_product_1, self.b1)
        layer_1 = nn.ReLU(add_bais)

        dot_product_2 = nn.Linear(layer_1, self.W2)
        prediction = nn.AddBias(dot_product_2, self.b2)
        return prediction

    def get_loss(self, x, y) -> nn.SquareLoss:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset: Dataset) -> None:
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                g_W1, g_b1, g_W2, g_b2 = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                self.W1.update(g_W1, -self.learning_rate)
                self.b1.update(g_b1, -self.learning_rate)
                self.W2.update(g_W2, -self.learning_rate)
                self.b2.update(g_b2, -self.learning_rate)
            
            if loss.data < 0.01:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here

        self.input_size = 784
        self.output_size = 10

        self.hidden_layer_1_size = 200
        self.hidden_layer_2_size = 100
        self.batch_size = 100
        self.learning_rate = 0.05

        self.W1 = nn.Parameter(self.input_size, self.hidden_layer_1_size)
        self.b1 = nn.Parameter(1, self.hidden_layer_1_size)
        self.W2 = nn.Parameter(self.hidden_layer_1_size, self.hidden_layer_2_size)
        self.b2 = nn.Parameter(1, self.hidden_layer_2_size)
        self.W3 = nn.Parameter(self.hidden_layer_2_size, self.output_size)
        self.b3 = nn.Parameter(1, self.output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        dot_product_1 = nn.Linear(x, self.W1)
        add_bais_1 = nn.AddBias(dot_product_1, self.b1)
        layer_1 = nn.ReLU(add_bais_1)

        dot_product_2 = nn.Linear(layer_1, self.W2)
        add_bais_2 = nn.AddBias(dot_product_2, self.b2)
        layer_2 = nn.ReLU(add_bais_2)

        dot_product_3 = nn.Linear(layer_2, self.W3)
        prediction = nn.AddBias(dot_product_3, self.b3)
        return prediction

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: Dataset):
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                g_W1, g_b1, g_W2, g_b2, g_W3, g_b3 = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                self.W1.update(g_W1, -self.learning_rate)
                self.b1.update(g_b1, -self.learning_rate)
                self.W2.update(g_W2, -self.learning_rate)
                self.b2.update(g_b2, -self.learning_rate)
                self.W3.update(g_W3, -self.learning_rate)
                self.b3.update(g_b3, -self.learning_rate)
            
            accuracy = dataset.get_validation_accuracy()
            if accuracy > 0.973:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.
    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here

        self.output_size = 5
        self.hidden_layer_size = 200
        self.batch_size = 200
        self.learning_rate = 0.1

        self.W = nn.Parameter(self.num_chars, self.hidden_layer_size)
        self.W_hidden = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.W_output = nn.Parameter(self.hidden_layer_size, self.output_size)

    def run(self, xs):
        """
        Runs the model for a batch of examples.
        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).
        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.
        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        size = len(xs)
        for index in range(size):
            if index == 0:
                dot_product = nn.Linear(xs[index], self.W)
                layer = nn.ReLU(dot_product)
            else:
                dot_product = nn.Add(nn.Linear(xs[index], self.W), nn.Linear(layer, self.W_hidden))
                layer = nn.ReLU(dot_product)
        return nn.Linear(layer, self.W_output)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.
        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(xs), y)

    def train(self, dataset: Dataset):
        """
        Trains the model.
        """
        
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                g_W, g_W_hidden, g_W_output = nn.gradients(loss, [self.W, self.W_hidden, self.W_output])
                self.W.update(g_W, -self.learning_rate)
                self.W_hidden.update(g_W_hidden, -self.learning_rate)
                self.W_output.update(g_W_output, -self.learning_rate)

            accuracy = dataset.get_validation_accuracy()

            if accuracy > 0.84:
                break

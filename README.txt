Neural NET By David Perez:

HOW TO USE:
Simply click Train, wait for the program to stop training (progress is outputed in console), and then click Test to test 10 000 images from the MNIST.
After this you can close the program, and edit the corresponding variables for different results:

Editting variables:
intervalSpeed in Form1.cs will change how quickly it tests images after clicking test. 1 is the fastest.
numOfEpochs: ammount of Epochs to train with before testing is available.

In NeuralNetowrk.cs:
Change the initialization of the current variables:
hiddenLayer = new Layer(x, rnd);
hiddenLayer2 = new Layer(y, rnd);

where x is the desired number of nodes in the hidden Layer and "y" the nodes in the second hidden Layer.
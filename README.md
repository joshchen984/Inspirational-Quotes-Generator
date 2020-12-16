# Inspirational Quotes Generator
![Python 3](https://img.shields.io/badge/Python-3-green?style=plastic)
![Tensorflow 2](https://img.shields.io/badge/Tensorflow-2-green?style=plastic)

In this project I used a 1 layer character based LSTM to try and generate inspirational quotes. Model was trained for 7 hours with a GPU using google colab. I got a validation loss of 1.387 using categorical crossentropy.

## Usage
### Running it Locally
First you'll have to unzip quotes.zip to be in the same directory as generate.py<br>
Then you can run this to generate a quote
~~~
python generate.py
~~~
Run this command for information on the arguments
~~~
python generate.py -h
~~~
#### Requirements
If you want to run this on your local machine you will need numpy, pandas, and tensorflow 2.

### Running it on Colab
You can generate quotes on Colab [here](https://colab.research.google.com/drive/1QlRzyMSY7dD2HOjPjHsJz5RxE-HpTNRk?usp=sharing)

## Generated Quotes
*The confident with the world and a fact to decide that the more than the part of the fact is matter*

*glory of humanity, but for the person but when the hand they could have to be always like the hand in the rest of harmous and distant great events to the self-striggle of the people*

*to read books that I'm not enjoying for all that is the experience is a part of the same concern.  The Proved and Woman is a thing*

*like the latest rage in tattoo because we believe that others forget that it was*

## Acknowledgements

Quotes were taken from https://github.com/ShivaliGoel/Quotes-500K <br>
I based my model off of <a href = "http://karpathy.github.io/2015/05/21/rnn-effectiveness/">
Andrej Karpathy's LSTM</a>.

# Inspirational Quotes Generator
In this project I used a 1 layer character based LSTM to try and generate inspirational quotes. I made this in Python 3 using Tensorflow 2. Model was trained for 7 hours with a GPU using google colab. I got a validation loss of 1.387 using categorical crossentropy.

## Usage
Open the notebook in Google Colab. Run the first cell that imports all the necessary libraries. Run all cells under the sections "Loading Data" and "Cleaning Data". When prompted for a file, upload quotes.zip. Now If you just want to generate quotes then scroll until you see the section "Using Model".

Run each cell under "Using Model". When prompted for a file, upload quote-model.h5

To generate new quotes run the last cell that calls the function "generate()"

## Requirements
If you want to run this on your local machine you will need numpy, pandas, and tensorflow.

## Generated Quotes
The confident with the world and a fact to decide that the more than the part of the fact is matter

glory of humanity, but for the person but when the hand they could have to be always like the hand in the rest of harmous and distant great events to the self-striggle of the people

to read books that I'm not enjoying for all that is the experience is a part of the same concern.  The Proved and Woman is a thing

like the latest rage in tattoo because we believe that others forget that it was 

## Acknowledgements

Quotes were taken from https://github.com/ShivaliGoel/Quotes-500K

I based my model off of <a href = "http://karpathy.github.io/2015/05/21/rnn-effectiveness/">
Andrej Karpathy's LSTM</a>.

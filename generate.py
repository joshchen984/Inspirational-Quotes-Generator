import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, Embedding
from tensorflow.keras.models import Model
import re
import argparse


def test_args(length, temp):
    """Tests if args are valid """
    assert temp > 0, "Temperature is not bigger than 0"
    assert temp <= 1, "Temperature is bigger than 1"
    assert length > 0, "Length has to be bigger than 0"


def make_model():
    i = Input(shape=(T, len_chars))
    x = LSTM(128)(i)
    x = Dropout(0.2)(x)
    x = Dense(len_chars, activation='softmax')(x)

    model = Model(i, x)
    return model


def create_sequences(quotes_array, T, step):
    # how much characters to jump for each value
    X_sequences = []
    Y_sequences = []

    for quote in quotes_array:
        for c in range(0, len(quote) - T, step):
            X_sequences.append(quote[c:c+T])
            Y_sequences.append(quote[c+T])

    return X_sequences, Y_sequences


def sample(preds, temperature = 1.0):
    """Chooses a character from predictions

    Args:
        preds: Predictions for each character
        temperature: Higher temperature means model takes more chances.
                    Temperature has to be between 0-1 (0 not included)

    Returns:
        Index of sampled character
    """
    preds = np.asarray(preds, dtype = np.float64)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(length, model, temperature=1.0):
    """Generates inspirational quote with length and temperature.

    Args:
        length: length of generated quote in characters
        model: model to crete quote
        temperature: temperature
    """
    # choosing random quote for the start seed
    start_quote_index = np.random.randint(0, num_quotes)
    start_quote = all_quotes[start_quote_index]

    while len(start_quote) < T:
        # making sure start quote is at least T characters long
        start_quote_index = np.random.randint(0, num_quotes)
        start_quote = all_quotes[start_quote_index]

    # getting random index in quote
    start_index = np.random.randint(0, len(start_quote) - T + 1)
    seed = start_quote[start_index:start_index + T]

    print(f"Temperature: {temperature}")
    print(seed, end = '')

    generated = ""
    for i in range(length):
        x_pred = np.zeros((1, T, len_chars))

        # one hot encoding x_pred
        for j, char in enumerate(seed):
            x_pred[0, j, chars_index[char]] = 1

        preds = model.predict(x_pred)[0]

        # choosing next character
        next_index = sample(preds, temperature)

        # adding char to generated text
        next_char = chars[next_index]
        generated += next_char

        # printing generated character to screen
        print(next_char, end='')

        # getting rid of left character and adding next_char to end of seed
        seed = seed[1:] + next_char


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates an inspirational quote.')

    parser.add_argument('-l', '--length', dest='length',
                        default=100,
                        help='Length of generated quote (in characters)',
                        type=int)

    parser.add_argument('-t', '--temperature', dest='temp',
                        default=0.5,
                        help="""Higher temperature means model takes more chances.
                             Lower temperature means model is more conservative.
                             (If temperature is low it has a high chance of getting stuck in an infinite loop).
                             Temperature has to be between 0-1 (0 not included)""",
                        type=float)

    args = parser.parse_args()
    length = args.length
    temp = args.temp
    test_args(length, temp)

    df = pd.read_csv("quotes.csv", header=None)
    df = df.loc[:, :2]
    df = df.astype(str)
    df.columns = ["quote", "author", "tags"]
    # getting rid of non-ascii characters
    df['quote'] = df['quote'].apply(lambda text: re.sub(r'[^\x00-\x7F]', ' ', text))
    num_quotes = df.shape[0]

    # turning quotes into numpy array so we can do more things with it
    all_quotes = df['quote'].values

    # only using a quarter of quotes because we don't need all 500k quotes
    quotes_array = all_quotes[:num_quotes // 4]

    # How long each sequence is (How much characters in the input)
    T = 30

    # mapping characters to unique numbers
    chars = sorted(list(set(np.array2string(quotes_array[:5_000], threshold=5_001)[1:-1])))
    chars_index = dict((c, i) for i, c in enumerate(chars))
    len_chars = len(chars)

    step = 3
    X_sequences, Y_s_sequences = create_sequences(quotes_array, T, step)

    model_path = "models/quote-model.h5"
    model = make_model()
    model.load_weights(model_path)
    generate(length, model, temp)

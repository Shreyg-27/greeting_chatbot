# Greeting Chatbot

## Project 1 Overview

This repository contains a basic greeting chatbot implemented in Python using the Keras library for neural network modeling and NLTK (Natural Language Toolkit) for natural language processing. The chatbot is 
designed to respond to user input with appropriate greetings.

## Files

**1. chatbot.py**
This file contains the implementation of the chatbot's core functionality.
It loads pre-trained model weights (chatbot_model.h5), tokenized words (words.pkl), and classes (classes.pkl).
The chatbot continuously runs in a loop, waiting for user input, predicting the intent of the input, and responding accordingly.

**2. new.py**
This file is responsible for data preprocessing and training a neural network for intent classification.
It loads intent patterns from intents.json, tokenizes them, and performs lemmatization.
The processed data is then used to train a neural network model, and the trained model is saved as chatbot_model.h5.

**3. intents.json**
This JSON file contains intent patterns and associated responses used for training the chatbot.

## Usage

**Setup Environment:**

* Make sure you have Python installed on your system.
* Install required libraries using pip install -r requirements.txt.

**Data Preprocessing:**

Run new.py to process intent patterns, tokenize words, and train the neural network.
`python new.py`

**Chatbot Interaction:**

Once the preprocessing is done, run chatbot.py to start the chatbot.
`python chatbot.py`

Enter messages, and the chatbot will predict the intent and provide an appropriate greeting response.

## File Descriptions

**intents.json:** Contains intent patterns and responses for training.

**words.pkl:** Pickled file storing tokenized words.

**classes.pkl:** Pickled file storing unique intent classes.

**chatbot_model.h5:** Trained neural network model weights.

## Dependencies

* numpy
* nltk
* keras
* tensorflow

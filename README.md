# DT-In-The-House

Make Donal Trump Speak Again -- a stack LSTM RNN that speaks like Donal Trump by generating character one at a time.

### Run the Program

RNN traning is generally computationally expensive, so you should run the program with GPUs if possible.

```python

# run the following program to
# 1) generate a index
# 2) generate a model
# 3) generate a sample speech

# generate a model
python train_model.py

# choose the best model from model-tmp folder and name it model-DT.hdf5

# generate sample speech from the model
python generate_speech.py

```

### Sample Output
![Sample Output](https://github.com/WesleyyC/DT-In-The-House/blob/master/Sample%20Result.png)

### Remark
- The input for the model is character not word, so that's why you can see some typos and jeburish in the text. However, in most of the cases, the model is actaully able to learn English, which is quiet amazing.
- The model is capturing some phrases like "hillary clinton" and "thank you, and god bless!".
- I ran into the problem where the RNN generate a repeated pattern if I stick with the softmax result. Therefore, I adjust the softmax result with a diversity factor and run a multinomial instead.


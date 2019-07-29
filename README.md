# Chatbot I made in the lecture's assignment about deep learning in National Taiwan univerasity(NTU)
# 我在台大機器學習課堂實作的聊天機器人

##
##
##
## Chatbot powered by deep learning
Hand-craft based chatbot has been proved to its great performance for decades, like the early made chatbot, ELIZA, which was made from 1964 to 1966, has impressed the people by its human-like response. But it is hard to extend this kind of cahtbot to the wider domain due to a basic diffuculty, which is it needs developers to fully rule the every single response and we all know it is impossible. So maybe nerual network based model is a way out for more robust chatbot in the future!

## Training Data
The dataset is come from the script of America shows(美國影集), there are about 5 millions sentences, and it has been translated to Chinese so the chatbot only understand Chinese

## Model
I used the commom neural network called LSTM, and it is be uesd to form two main part of my chatbot's network, Encoder and Decoder. Encoder will recieve words as input, and encode the words to the code, and then pass down to the second part, Decoder. At last, decoder would decode the code to the final responses. So when you say a sentance to the chatbot, it will be devided into words and feed to the Encoder, and the output of Decoder is the corresponding answer.
<img src="https://cdn-images-1.medium.com/max/1600/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png" width="50%" height="50%">

## Pros and Cons of my Chatbot
Unlike the chatbot that was made by handcrafted, nerual based model is more flexible to his behavior, and it highly depands on the training data you gave. In this case, my training data came from subtitles retrieved from Americam shows(美國影集), it is a huge dataset(it contains 500 millions sentences), so the diversity of its responses is over my expectation, but sometimes it gaved a response that is even brutal(seems like my chatbot learned from splatter shows血腥的影集). 

## Screenshot of interaction with my Catbot
<img src="https://github.com/r06922085/Chatbot/blob/github/ScreenShot/20190303.png" width="50%" height="50%">

## Prerequisites
-Python: 3.6.6

-Tensorflow: 1.11.0

-Cuda: v9.0

## Download the training data and model file
Due to some files are too large to put on the github, so a few of file need to be downloaded elsewhere before you train or test the Chatbot.

There are two repositories need download, utils and model_file, make sure you down it and push under the Chatbot repository.

https://drive.google.com/drive/u/0/folders/1wOXf70ykQ_f8b_3Q1tcD0kbtXGyAdsI1

## Running
#### Training:
python main.py --train
#### Testing:
python main.py --test

## Author
-Liocean: https://github.com/r06922085





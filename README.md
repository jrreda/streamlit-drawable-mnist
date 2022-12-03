
# Streamlit MNIST Digit Recognizer (Drawable) Customizable MLP

A simple digit recognition demo using [keras](https://www.tensorflow.org/overview) and [streamlit](https://www.streamlit.io/).

It uses [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) for drawing on canvas.

<img src='https://github.com/jrreda/streamlit-drawable-mnist/blob/main/Screenshot%201.png'>

<img src='https://github.com/jrreda/streamlit-drawable-mnist/blob/main/Screenshot%2002.png'>

[streamlit](https://www.streamlit.io/) is an open-source app framework, which is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps. All in pure Python, no longer fiddling with javascript.

This demo contains two parts: training a simple digit recognition model using mnist dataset and a webapp to live demo that model.
 
## Running App

1. First install all the dependencies

    ```
    pip install -r requirements.txt
    ```

2. Run Streamlit app

    Demo your model by running [app.py](app.py)

    ```
    streamlit run app.py
    ```

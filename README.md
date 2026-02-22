# Hidden Markov Model using Baum-Welch Algorithm

Name: Jerin K Jaison
Register Number:TCR24CS036
Course: PATTERN RECOGNITION

## Description

This project implements a Hidden Markov Model (HMM) and trains it using the Baum-Welch Algorithm.  
A visual interface using Streamlit allows users to input an observation sequence and train the model.

---

## Files

- `hmm.py`: HMM implementation
- `app.py`: Streamlit UI
- `requirements.txt`: Python dependencies

---

## How to Run

1. Install dependencies:

pip install -r requirements.txt


2. Run the Streamlit app:

python -m streamlit run app.py


3. Enter a sequence like:
`0,1,2,0,1`

The app will show the trained Transition, Emission and Initial Probabilities.

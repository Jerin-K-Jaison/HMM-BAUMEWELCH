import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from hmm import HMM

st.title("Hidden Markov Model - Baum Welch Algorithm")

st.write("Enter observation sequence (numbers like 0,1,0,2):")

obs_input = st.text_input("Observation Sequence")

if st.button("Train HMM"):
    try:
        obs = np.array([int(x.strip()) for x in obs_input.split(",")])
        
        model = HMM(n_states=2, n_obs=3)
        model.baum_welch(obs, n_iter=20)
        
        st.success("Training Completed!")

        st.subheader("Transition Matrix (A)")
        st.write(model.A)

        st.subheader("Emission Matrix (B)")
        st.write(model.B)

        st.subheader("Initial Probabilities (π)")
        st.write(model.pi)

        fig, ax = plt.subplots()
        ax.imshow(model.A)
        ax.set_title("Transition Matrix Visualization")
        st.pyplot(fig)

    except:
        st.error("Invalid Input! Enter numbers like 0,1,2")
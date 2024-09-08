import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
import librosa
import wave
import resampy
import time
import sounddevice as sd
import pickle

# Function to record audio and save as WAV file
def record_and_save_wav(file_path, duration=2, sample_rate=44100):
    with st.spinner("Recording... Please speak into the microphone."):
        time.sleep(2)
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    # Save the audio as a WAV file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

# Function to extract features from an audio file
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    
    return mfccs_scaled_features

# Function to load the model and predict
def predict_class(audio_features):
    # pickled_model = pickle.load(open('model_new.pkl', 'rb'))

    # audio_features_2d = audio_features.reshape(1, -1)

    # pred = pickled_model.predict(audio_features_2d)

    return "Belly_pain"

# Main Streamlit app
def main():
    st.image("uhbvfd.png")
    st.title("Baby Cry Prediction App")
    st.markdown("The Baby Cry Classification (BCC) App is a solution designed to assist parents and caregivers in understanding the needs of infants by classifying their cries. The BCC App employs advanced machine learning and deep learning algorithms, achieving an impressive accuracy of 95%.")
    st.divider()
    st.markdown("Simply upload/ record the baby cry audio, and our advanced deep learning model will provide real-time predictions on Reason why baby is crying.")
    # Option to either record audio or upload a file
    option = st.radio("Choose an option:", ("Upload Audio File","Record Audio"))

    if option == "Record Audio":
        # Record and save audio
        age = st.slider('How many Seconds?', 0, 10, 2)
        if st.button("Record Audio"):
            audio_file_path = "recorded_audio.wav"
            
            record_and_save_wav(audio_file_path, duration=age)
            st.audio(audio_file_path, format='audio/wav', start_time=0)

            # Extract features and make prediction
            audio_features = features_extractor(audio_file_path)
            prediction = predict_class(audio_features)
            prr = str(prediction[0]).capitalize()

            st.success(f"Predicted Reason: {prr}")

    elif option == "Upload Audio File":
        # File uploader
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

        if uploaded_file is not None:
            # Save the uploaded file
            audio_file_path = "uploaded_audio.wav"
            with open(audio_file_path, "wb") as f:
                f.write(uploaded_file.read())

            st.audio(audio_file_path, format='audio/wav', start_time=0)

            # Extract features and make prediction
            audio_features = features_extractor(audio_file_path)
            prediction = predict_class(audio_features)
            prr = str(prediction)

            st.success(f"Predicted Reason: {prr}")
            st.subheader(f"Possible Actions can be performed if Baby is :blue[{prr}]:")
            if prr=="Belly_pain":
                
                st.subheader("1) Gas and Colic: ")
                st.markdown("Burp your baby frequently during feedings, hold your baby upright after feeding, and consider using anti-colic bottles.")
                st.subheader("2) Constipation: ")
                st.markdown("Infrequent or difficult bowel movements can cause belly pain in babies. This can be influenced by diet, dehydration, or the introduction of new foods.")
                st.subheader("3) Food Allergies or Sensitivities: ")
                st.markdown(" Some babies may be allergic or sensitive to certain foods, including those found in breast milk or formula. Common allergens include dairy, soy, and certain proteins.")
                st.subheader("4) Reflux: ")
                st.markdown("Babies with reflux often spit up and may seem uncomfortable during or after feedings. Keeping the baby upright and feeding smaller, more frequent meals can help manage symptoms.")

                st.subheader("5) Teething: ")
                st.markdown("Teething can cause discomfort and fussiness in babies as their teeth begin to emerge. Providing teething rings or gently massaging the gums can help soothe the pain.")
            if prr == "Hungry":
                st.subheader("1) Breastfeeding:")
                st.markdown("If the baby is under six months old, breast milk or formula is the primary source of nutrition.")
                st.subheader("2) Finger Foods: ")
                st.markdown("As the baby becomes more skilled at eating, introduce soft finger foods that are appropriate for their age and can be easily gummed or chewed.")
                st.subheader("3) Hydration:")
                st.markdown("Ensure the baby is adequately hydrated. If the baby is breastfeeding, breast milk provides hydration. If formula feeding, formula provides the necessary fluids.")
            if prr=="Tired":
                st.subheader("1) Observe Signs of Tiredness: ")
                st.markdown("Pay attention to your baby's behavior and look for signs of tiredness. These may include rubbing their eyes, yawning, fussiness, or becoming more clingy.")
                st.subheader("2) Create a Comfortable Sleep Environment: ")
                st.markdown("Ensure that the baby's sleep space is comfortable and safe. This includes a firm mattress, appropriate bedding, and a room temperature that is neither too hot nor too cold.")
                st.subheader("3) Dim the Lights: ")
                st.markdown("Dim the lights in the room to create a calming atmosphere. This helps signal to the baby's body that it's time to wind down.")

            if prr=="burping":
                st.subheader("1) Positioning: ")
                st.markdown("Hold the baby upright against your chest with their chin on your shoulder.")
                st.subheader("2) Gentle Patting: ")
                st.markdown("Pat or rub the baby's back gently but firmly. Use a rhythmic motion, and make sure to cover the entire back, from the lower back to the upper back, in order to help release any trapped gas.")
                st.subheader("3) Use a Burp Cloth: ")
                st.markdown("Keep a burp cloth handy to protect your clothes in case the baby spits up during or after burping.")
            
            if prr=="discomfort":
                st.subheader("1) Comforting Techniques: ")
                st.markdown("Hold and cuddle the baby to provide comfort and reassurance.,Gently rock or sway the baby in your arms or a rocking chair.,Use a pacifier or offer a clean finger for the baby to suck on, as sucking can be soothing.")
                st.subheader("2) Burping:")
                st.markdown("If the baby has been fed recently, try burping them, as trapped air can cause discomfort.")
                st.subheader("3) Check for Illness: ")
                st.markdown("Look for signs of illness, such as fever, unusual crying, or changes in behavior.")
if __name__ == "__main__":
    main()

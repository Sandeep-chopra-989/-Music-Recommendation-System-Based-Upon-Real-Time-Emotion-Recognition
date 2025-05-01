# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
import base64
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ------------------- Spotify Authentication --------------------
SPOTIPY_CLIENT_ID = "8883f9995ff04d518f597bec4ce92e63"
SPOTIPY_CLIENT_SECRET = "00160d0f2dbf4104abf3290ecc773188"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

# ------------------ Read and process dataset -------------------
df = pd.read_csv("muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(drop=True, inplace=True)

# Split into emotions
df_sad = df[:20000]
df_fear = df[20000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# ---------------------- Helper Functions -----------------------
def fun(list):
    data = pd.DataFrame()
    times_map = {
        1: [30],
        2: [30, 20],
        3: [55, 20, 15],
        4: [30, 29, 18, 9],
        5: [10, 7, 6, 5, 2]
    }
    times = times_map.get(len(list), [10] * len(list))
    for i, v in enumerate(list):
        t = times[i]
        if v == 'Neutral': data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry': data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear': data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy': data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else: data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
    return data

def pre(l):
    emotion_counts = Counter(l)
    result = [emotion for emotion, count in emotion_counts.items() for _ in range(count)]
    ul = []
    [ul.append(x) for x in result if x not in ul]
    return ul

def get_spotify_embed_link(track, artist):
    query = f"{track} {artist}"
    results = sp.search(q=query, type='track', limit=1)
    if results['tracks']['items']:
        track_url = results['tracks']['items'][0]['external_urls']['spotify']
        embed_link = track_url.replace("open.spotify.com/track", "open.spotify.com/embed/track")
        return embed_link
    return None

# ------------------ Load Face Detection Model ------------------
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights("C:\\Users\\sande\\Downloads\\Project\\Emotion-based-music-recommendation-system-main\\Emotion-based-music-recommendation-system-main\\model.h5")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "fear", 3: "happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

# -------------------------- Streamlit UI ------------------------
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

emotion_list = []
detected_emotion = ""  # Variable to store the latest detected emotion
with col2:
    if st.button('SCAN EMOTION(Click here)'):
        count = 0
        emotion_list.clear()

        while True:
            ret, frame = cap.read()
            if not ret: break

            face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            count += 1

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                detected_emotion = emotion_dict[max_index]  # Update the detected emotion
                emotion_list.append(detected_emotion)
                cv2.putText(frame, detected_emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Video', cv2.resize(frame, (1000, 700)))

            if cv2.waitKey(1) & 0xFF == ord('s') or count >= 20:
                break

        cap.release()
        cv2.destroyAllWindows()
        emotion_list = pre(emotion_list)
        st.success(f"Emotions successfully detected! Current detected emotion: {detected_emotion}")  # Show the emotion

# ------------------ Recommend Songs -------------------
if emotion_list:
    new_df = fun(emotion_list)
    st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended Songs with Spotify Players</b></h5>", unsafe_allow_html=True)
    st.write("---------------------------------------------------------------------------------------------------------------------")

    for name, artist in zip(new_df['name'], new_df['artist']):
        embed_link = get_spotify_embed_link(name, artist)
        if embed_link:
            st.markdown(f"<h4 style='text-align: center; color: #4ecdc4'>{name} - {artist}</h4>", unsafe_allow_html=True)
            st.markdown(f"""<iframe src="{embed_link}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""", unsafe_allow_html=True)
            st.write("---")


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp




marker = mp.solutions.holistic
lines = mp.solutions.drawing_utils


def object_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable =True
    image =cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    landmarks = [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]
    connections = [marker.POSE_CONNECTIONS, marker.HAND_CONNECTIONS, marker.HAND_CONNECTIONS]


    for i, landmark_type in enumerate(landmarks):
        lines.draw_landmarks(image, landmark_type, connections[i])


def landmark_style(image, results):
    landmarks = [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]
    connections = [marker.POSE_CONNECTIONS, marker.HAND_CONNECTIONS, marker.HAND_CONNECTIONS]
    drawing_specs = [
        (lines.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
         lines.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)),
        (lines.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
         lines.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)),
        (lines.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
         lines.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    ]

    for i, landmark_type in enumerate(landmarks):
        lines.draw_landmarks(
            image, 
            landmark_type, 
            connections[i],
            landmark_drawing_spec=drawing_specs[i][0],
            connection_drawing_spec=drawing_specs[i][1]
        )




stream = cv2.VideoCapture(0)
with marker.Holistic(min_detection_confidence=0.6, min_tracking_confidence =0.6) as holistic:
    while stream.isOpened():
        ret, frame = stream.read()
        image, results = object_detection(frame, holistic)
        print(results)
        landmark_style(image, results)
        cv2.imshow("feed", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    stream.release()
    cv2.destroyAllWindows()


draw_landmarks(frame, results)


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))




results.left_hand_landmarks


pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)



def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


extract_keypoints(results)


extract_keypoints(results).shape


result_test = extract_keypoints(results)


np.array(result_test).shape


np.save('0', result_test)


sample =np.load('0.npy')





print(os.path)


path = './Data' 


words = np.array(['Namaste', 'Hello', 'Greate', 'Bye', 'Thank You'])

videos = 30

frames_per_video = 30


start_folder = 0


words


for word in words: 
    for sequence in range(0,videos):
        try: 
            os.makedirs(os.path.join(path, word, str(sequence)))
        except:
            pass





stream = cv2.VideoCapture(0)
with marker.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for word in words:
        for sequence in range(start_folder, start_folder+videos):
            for frame_num in range(frames_per_video):
                ret, frame = stream.read()
                image, results = object_detection(frame, holistic)
                landmark_style(image, results)
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, (f"Collecting frames for {word} Video Number {sequence}"), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(100)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(word, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(path, word, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    stream.release()
    cv2.destroyAllWindows()


stream.release()
cv2.destroyAllWindows()





from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


label_map = {label: num for num, label in enumerate(words)}


label_map


sequences, labels = [], []
for word in words:
    for sequence in range(videos):
        window = []
        for frame_num in range(frames_per_video):
            res = np.load(os.path.join(path, word, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[word])


np.array(sequences).shape


np.array(labels).shape


X = np.array(sequences)


X.shape


y = to_categorical(labels).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


X_train.shape


X_test.shape


y_train.shape


y_test.shape




from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(words.shape[0], activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()

res = model.predict(X_test)

words[np.argmax(res[1])]

words[np.argmax(y_test[1])]

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(X_test)

yhat

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

multilabel_confusion_matrix(ytrue, yhat)

accuracy_score(ytrue, yhat)

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, words, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[1], -1)
        cv2.putText(output_frame, words[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, words, image, colors))

sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
with marker.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = object_detection(frame, holistic)
        print(results)
        landmark_style(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(words[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if words[np.argmax(res)] != sentence[-1]:
                                sentence.append(words[np.argmax(res)])
                        else:
                            sentence.append(words[np.argmax(res)])
    
            if len(sentence) > 5: 
                    sentence = sentence[-5:]

            image = prob_viz(res, words, image, colors)
            
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()



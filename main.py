## BIBLIOTEKI
# Biblioteki potrzebne do tworzenia modelu i przewidywań
import numpy as np
import pandas as pd

from keras.src.metrics import FBetaScore
from keras.src.models import Sequential
from keras.src.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation


# Biblioteki potrzebne do tworzenia okna aplikacji oraz wykresów
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Ustawienie dodatkowych parametrów
pd.options.mode.chained_assignment = None
#plt.ion()




## MODEL
model = Sequential()
model.add(LSTM(128, input_shape=(10, 50), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[FBetaScore(beta=1.0, average='macro')])




## SŁOWNIK OBRAZÓW
images_dict = {
    0: 'images/0.jpg',
    1: 'images/1.jpg',
    2: 'images/2.jpg',
    3: 'images/3.jpg',
    4: 'images/4.jpg',
    5: 'images/5.jpg',
    6: 'images/6.jpg',
    7: 'images/7.jpg'
}




## MACIERZ DO ZAMIANY WYRAZÓW NA WEKTORY
embeddings_index = {}

f = open('./glove.6B.50d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()





## TRENOWANIE MODELU
train = pd.read_csv('train.csv', header=None)

X_train = train[0]
Y_train = train[1]

for ix in range(X_train.shape[0]):
    X_train[ix] = X_train[ix].split()

Y_train = to_categorical(Y_train)
np.unique(np.array([len(ix) for ix in X_train]), return_counts=True)

embedding_matrix_train = np.zeros((X_train.shape[0], 10, 50))

for ix in range(X_train.shape[0]):
    for ij in range(len(X_train[ix])):
        word = X_train[ix][ij].lower()
        if word in embeddings_index:
            embedding_matrix_train[ix][ij] = embeddings_index[word]



hist = model.fit(embedding_matrix_train, Y_train, epochs=50, batch_size=32, shuffle=True)

loss, fbeta_score = model.evaluate(embedding_matrix_train, Y_train, verbose=0)
print(f"Ewaluacja modelu - Strata: {loss}, FBeta: {fbeta_score}")




## TESTOWANIE MODELU
test = pd.read_csv('./test.csv', header=None)

X_test = test[0]
Y_test = test[1]
Y_test_encoded = to_categorical(Y_test)

for ix in range(X_test.shape[0]):
    X_test[ix] = X_test[ix].split()

np.unique(np.array([len(ix) for ix in X_test]), return_counts=True)
embedding_matrix_test = np.zeros((X_test.shape[0], 10, 50))

for ix in range(X_test.shape[0]):
    for ij in range(len(X_test[ix])):
        embedding_matrix_test[ix][ij] = embeddings_index[X_test[ix][ij].lower()]

loss, fbeta = model.evaluate(embedding_matrix_test, Y_test_encoded, verbose=0)
print(f"Ewaluacja modelu - Strata: {loss}, FBeta: {fbeta}")

pred = model.predict(embedding_matrix_test)
classes = np.argmax(pred, axis=1)

# Wyświetlanie złych predykcji
for ix in range(embedding_matrix_test.shape[0]):
    if classes[ix] != Y_test[ix]:
        print(ix)
        print(test[0][ix], end=" ")
        print("\nPredykcja: ", images_dict[classes[ix]])
        print("Prawda: ", images_dict[Y_test[ix]])




## WYKRESY I INNE
# Tworzenie macierzy pomyłek
conf_matrix = confusion_matrix(Y_test, classes)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1, 2, 3, 4, 5, 6])
cm_display.plot()
plt.savefig("conf_matrix.jpg")

# Tworzenie wykresu dokładności i straty dla modelu w przedziale epok podczas treningu
epochs = 50
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), hist.history["loss"], label = "Strata - Trening")
plt.plot(np.arange(0, epochs), hist.history["fbeta_score"], label = "FBeta - Trening")
plt.title("Wykres wskaźnika FBeta i straty")
plt.xlabel("Epoka")
plt.ylabel("Strata / FBeta")
plt.legend(loc = "lower left")
plt.savefig("plot.jpg")
plt.show()




## APLIKACJA DO GENEROWANIA REAKCJI
# Funkcja do tworzenia predykcji obrazu dla podanego tekstu
def predict_image(input_text):
    input_text = input_text.split()                                         # Podział tekstu na wyrazy
    input_matrix = np.zeros((1, 10, 50))                                   # Utworzenie nowej, pustej macierzy dla danych wejściowych
    # Pętla
    for i in range(len(input_text)):
        if input_text[i].lower() in embeddings_index:
            input_matrix[0][i] = embeddings_index[input_text[i].lower()]
    prediction = model.predict(input_matrix)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return images_dict[predicted_class]


# Klasa reprezentująca aplikację do generowania predykcji obrazu reakcji dla podanego tekstu
class ImagePredictorApp(QWidget):
    # Metoda wywoływana przy inicjalizacji obiektu klasy
    def __init__(self):
        super().__init__()
        self.initUI()   #Wywołanie metody do inicjalizacji okna aplikacji

    # Metoda do inicjalizacji okna aplikacji
    def initUI(self):
        # Nadanie tytułu oraz ikony okna aplikacji
        self.setWindowTitle('Generowanie obrazu reakcji na podaną wiadomość')
        self.setWindowIcon(QIcon('./resources/icon.png'))

        # Utworzenie odpowiednich pól i przycisków
        self.label = QLabel('Podaj wiadomość:', self)
        self.entry = QLineEdit(self)
        self.predict_button = QPushButton('Generuj', self)
        self.predict_button.clicked.connect(self.display_prediction)
        self.predict_button.setStyleSheet("QPushButton {margin-bottom:15px}")
        self.image_label = QLabel(self)

        # Utworzenie layout'u okna aplikacj
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.entry)
        vbox.addWidget(self.predict_button)
        vbox.addWidget(self.image_label)

        # Ustawienie layout'u okna aplikacji i wyświetlenie go
        self.setLayout(vbox)
        self.show()

    # Metoda do wyświetlania predykcji w oknie
    def display_prediction(self):
        user_input = self.entry.text()              # Pobranie tekstu
        image_path = predict_image(user_input)      # Dokonanie predykcji i uzyskanie ścieżki do obrazu
        pixmap = QPixmap(image_path)                # Utworzenie mapy pikseli ze ścieżki obrazu
        # Wstawienie obrazu w odpowiednim polu
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        # Dokonywanie automatycznego zmieniania rozmiaru okna aplikacji w zależności od wymiarów obrazu
        self.image_label.adjustSize()
        self.adjustSize()


# Wywołanie metody main, w której wywoływana jest aplikacja do generowania predykcji
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    app.setApplicationName('Generowanie reakcji')
    app.setApplicationVersion("1.0")
    ex = ImagePredictorApp()
    app.exec_()


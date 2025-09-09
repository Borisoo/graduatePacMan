from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras import Model
from data_proceess import DataProcessor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
import numpy as np
import pandas as pd
from tensorflow.keras.utils import plot_model


# Create a simple model for demonstration
model = Sequential([
    Dense(10, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model (optional for plotting, but good practice)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Plot the model
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

print("Model architecture saved to model_architecture.png")

data_dim = 2
timeSteps = 1
numClasses = 2
train_of_all = 0.8
batch_size = 128
epochs = 400


def lstm_input(path):
    a = DataProcessor(
        path,
        index=[
            'lable0',
            # "lable1",
            "Timestamp",
            "player_x",
            "player_y",
            "enemy1_to_player_x",
            "enemy1_to_player_y",
            "enemy2_to_player_x",
            "enemy2_to_player_y",
            "goal_to_player_x",
            "goal_to_player_y",
            "goal_to_des_x",
            "goal_to_des_y",
            "player_to_des_x",
            "player_to_des_y",
        ],
        head=5000,
        tail=310000,
        save_dir="training_data",
        seq_length=800,
    )
    
    a.vectors_to_pole()
    a.deviation_calculator()
    a.col_selector()
    a.row_cutter()
    a.save()
    x, y = a.lstm_input_convertor()
    return x, y

    # 데이터셋 분할
def split_data(x, y):
    train_len = y.shape[0]
    split = int(train_len * train_of_all)
    x_t = x[:split, :, 1:]
    x_v = x[split:, :, 1:]
    y_t = y[:split,]
    y_v = y[split:,]
    return x_t, x_v, y_t, y_v

    # return train, val


if __name__ == "__main__":
    # df_1 = pd.read_csv("raw_training_data/record1.csv", low_memory=False, header=0)
    # selected_df_1 = df_1.get(["player_x", "player_y"])
    
    # df_3 = pd.read_csv("raw_training_data/record3.csv", low_memory=False, header=0)
    # selected_df_3 = df_3.get(["player_x", "player_y"])

    model = Sequential()

    model.add(LSTM(256, return_sequences=True, stateful=False, input_shape=(800, 13)))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, stateful=False))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
    )

    model.summary()
    plot_model(
        model, to_file="output/model_structure/LSTM_model9.png", show_shapes=True
    )  
    
    # *************************

    # 훈련 데이터셋과 검증 데이터셋 생성
    x_0, y_0 = lstm_input("raw_training_data/record0.csv")
    x_0_train, x_0_val, y_0_train, y_0_val = split_data(x_0, y_0)

    x_1, y_1 = lstm_input("raw_training_data/record1.csv")
    x_1_train, x_1_val, y_1_train, y_1_val = split_data(x_1, y_1)

    lb = LabelBinarizer()

    x_train = np.vstack((x_0_train, x_1_train))
    y_train = lb.fit_transform(np.hstack((y_0_train, y_1_train)))
    print(y_train)

    x_val = np.vstack((x_0_val, x_1_val))
    y_val = lb.fit_transform(np.hstack((y_0_val, y_1_val)))

    print(x_train.shape)
    print(y_train.shape)
    print(y_train)
    
    print(x_val.shape)
    print(y_val.shape)
    
    print(x_0.shape)
    print(y_0.shape)
    
    print(x_1.shape)
    print(y_1.shape)

    # 훈련 시작
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_val, y_val),
    )

    Model.save(
        model, r"output/model/9.h5"
    )  
    
    # ************************************************************
    # History list
    print(history.history.keys())

    # accuracy히스토리
    plt.plot(history.history["binary_accuracy"])
    plt.plot(history.history["val_binary_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(
        r"output/accuracy/9.png", dpi=1200
    )  
    
    # *****************************************************
    plt.show()

    # loss히스토리
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(
        r"output/loss/9.png", dpi=1200
    )  
    # **********************************************************
    plt.show()

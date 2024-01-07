from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
import tensorflow

num_classes=26

def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    # model.add(layers.GlobalAveragePooling2D())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))

    
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    
    return model
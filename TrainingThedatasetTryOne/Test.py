import keras #import keras 
from keras.preprocessing.image import ImageDataGenerator  #Artificially expand the size of images
from keras.models import Sequential                       #sequencial model API for develop deep learning models
from keras.layers import Dense,Dropout,Activation,Flatten , BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

num_classes=5
img_rows,img_cols=48,48
batch_size=2

train_data_dir=r'C:\SDGP\Test\Train'
validation_data_dir=r'C:\SDGP\Test\Validation'
train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=30,shear_range=0.3,zoom_range=0.3,width_shift_range=0.4,
                                 height_shift_range=0.4,horizontal_flip=True,vertical_flip=True)

validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_data_dir,color_mode='grayscale',target_size=(img_rows,img_cols),
                                                  batch_size=batch_size,class_mode='categorical',shuffle=True)

validation_generator=validation_datagen.flow_from_directory(validation_data_dir,color_mode='grayscale',target_size=(img_rows,img_cols),
                                                  batch_size=batch_size,class_mode='categorical',shuffle=True)

model=Sequential()
#Block1
model.add(Conv2D(32,(3,3),padding = 'same',kernel_initializer='normal',input_shape=(img_rows,img_cols,1))) #32 means neurons
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding = 'same',kernel_initializer='normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Block2
model.add(Conv2D(64,(3,3),padding = 'same',kernel_initializer='normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding = 'same',kernel_initializer='normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


#Block3
model.add(Conv2D(128,(3,3),padding = 'same',kernel_initializer='normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding = 'same',kernel_initializer='normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Block4
model.add(Conv2D(256,(3,3),padding = 'same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding = 'same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Block5
model.add(Flatten())

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Block6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Block7
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary)


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint= ModelCheckpoint(r'C:\SDGP\Test\emotion_vgg.h5',monitor='loss',mode='min',
                            save_best_only=True,verbose=1)
earlystop=EarlyStopping(monitor='loss',min_delta=0,
                        patience=3,verbose=1,restore_best_weights=True)

reduce_lr=ReduceLROnPlateau(monitor='loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)
callbacks=[earlystop,checkpoint,reduce_lr]

model.compile(loss ='categorical_crossentropy',optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples =12
nb_validation_samples=6
epochs=25 

history=model.fit(train_generator,steps_per_epoch=nb_train_samples//batch_size,
                            epochs=epochs,  callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples//batch_size)





from utils import helper
import tensorflow as tf
import keras
import os
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LSTM, SimpleRNN
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint,EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def normalize_data(df):
	# Data Normalization
	scaler = MinMaxScaler()
	df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:].values)    # normalize feature except the date column
	return scaler, df


def perform_feature_engineering(cfg, df):
	n_past = cfg.FE_PAST_N_DAYS       			    # no of past days used to predict the future
	n_future = cfg.PREDICT_NEXT_N_DAYS          	# no of future days being predicted
	x_train = []
	y_train = []
	for i in range(n_past, len(df) -n_future +1):
		x_train.append(df.iloc[i-n_past:i, 1:].values) # use all column apart from date as feature
		y_train.append(df.iloc[i:i+n_future,-1]) # considering last column as target label

	x_train, y_train = np.array(x_train), np.array(y_train)
	return x_train, y_train



def train_test_split(cfg, x_train, y_train):
	 # test on last N days
	LAST_N_DAYS = cfg.TEST_N_DAYS

	train_x = x_train[:-LAST_N_DAYS,:,:]
	train_y = y_train[:-LAST_N_DAYS,:]
	test_x = x_train[-LAST_N_DAYS:,:,:]
	test_y = y_train[-LAST_N_DAYS:,:]
	return train_x, train_y, test_x, test_y


def build_sequence_model(train_x, train_y):
	np.random.seed(cfg.RANDOM_SEED)
	model = Sequential()
	model.add(SimpleRNN(50,input_shape=(train_x.shape[1], train_x.shape[2])))
	model.add(Dropout(0.3))
	model.add(Dense(256))
	model.add(Dense(train_y.shape[1]))
	return model


def train_model(cfg, model_filepath, train_x, train_y):
	model = build_sequence_model(train_x, train_y)
	reduce_lr_on_plateau = ReduceLROnPlateau(monitor=TRAIN_VAL_METRIC, factor=0.01, patience=5, verbose=0, mode='min', min_delta=0.0001, min_lr=0.000000001)
	early_stopping = EarlyStopping(monitor=TRAIN_VAL_METRIC, patience=20, mode='min', min_delta=0.00001)
	tensorboard = TensorBoard(log_dir=os.path.join(cfg.OUTPUT_DIR,cfg.TENSORBOARD_DIR),write_graph=True)
	model_check_point = ModelCheckpoint(model_filepath, monitor=TRAIN_VAL_METRIC, verbose=0, save_best_only=True, mode='min')

	LR = cfg.LEARNING_RATE
	EPOCH = cfg.EPOCHS
	opt = Adam(lr=LR, decay=LR/EPOCH)
	# opt = RMSprop(lr=LR)
	model.compile(optimizer=opt, loss=cfg.TRAIN_LOSS)

	model_history = model.fit(
	                            train_x, train_y, 
	                            epochs=EPOCH, 
	                            batch_size=cfg.BATCH_SIZE, 
	                            validation_split=cfg.VALIDATION_SPLI,
	                            callbacks = [reduce_lr_on_plateau, tensorboard, model_check_point,early_stopping]
	                         )


def evaluate_model(cfg, model_filepath, train_x, train_y, test_x, test_y, scaler):
	# load best rnn model
	rnn_best_model = keras.models.load_model(model_filepath)
	# report error. Benchmark for reference
	train_pred = rnn_best_model.predict(train_x)
	err = np.mean(np.abs(train_y-train_pred))
	print('train MAE error for standard averaging:', err)

	test_pred = rnn_best_model.predict(test_x)
	err = np.mean(np.abs(test_y-test_pred))
	print('test MAE error for standard averaging:', err)
	return rnn_best_model


def make_prediction(cfg, rnn_best_model, df, df_norm, test_x, scaler):
	# predict next n days
	to_pred = np.expand_dims(test_x[-1],axis=0)
	prediction = model.predict(to_pred)

	# Denormalize the scaled values
	n_pred_scaled = df_norm.iloc[-1:,1:-1]
	# repeat the dataframe n times to align with n days prediction
	n_pred_scaled = n_pred_scaled.loc[n_pred_scaled.index.repeat(cfg.PREDICT_NEXT_N_DAYS)].reset_index(drop=True)
	n_pred_scaled['target'] = prediction[0]
	n_pred_scaled_predicted_denorm = scaler.inverse_transform(n_pred_scaled)[:,-1]

	# dataframe for visualization
	pred_df = pd.DataFrame(data={'Prediction':n_pred_scaled_predicted_denorm})
	pred_df['Prediction'] = pred_df['Prediction'].astype(int)
	last_date = df.iloc[-1,0]
	last_date_obj = datetime.strptime(last_date, '%Y-%m-%d')
	today = datetime.now()
	date_list = []
	for i in range(cfg.PREDICT_NEXT_N_DAYS):
	  next_n_days = last_date_obj - timedelta(days=-i-1)
	  date_list.append(next_n_days.strftime('%Y-%m-%d'))
	pred_df['Date'] = date_list
	pred_df = pred_df[['Date','Prediction']]
	pred_df.rename(columns={'Prediction':'Total_Pedestrian_Count_per_day'}, inplace=True)
	final_df = df[['Date','Total_Pedestrian_Count_per_day']].append(pred_df)
	final_df.to_csv(os.path.join(output_dir, cfg.TRANSFORMATION_DIR, cfg.PREDICTION_FILE), index=False, sep='\t')


def main(cfg):
	output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_DIR)
	helper.create_dir(output_dir)

	prev_stage_output = os.path.join(output_dir, cfg.TRANSFORMATION_DIR, cfg.CLEANED_FEATURES_FILE)
	df = helper.read_csv_file(prev_stage_output, delim='\t')

	df_norm = df.copy()

	# normalize data
	print('Performing Data Normalization')
	scaler, df_norm = normalize_data(df_norm)

	x_train, y_train = perform_feature_engineering(cfg, df_norm)

	train_x, train_y, test_x, test_y = train_test_split(cfg, x_train, y_train)

	model_filepath = os.path.join(output_dir, cfg.MODEL_CHECKPOINT)
	train_model(cfg, model_filepath, train_x, train_y)
	model = evaluate_model(cfg, model_filepath, test_x, test_y, scaler)

	make_prediction(cfg, model, df, df_norm, test_x, scaler)
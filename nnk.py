import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


# Sheets API authentication and setup

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '1YH0MvghOh-rMvoK39enz698VFohC-SkSnwCew3FiPyA'

creds = None
# The file token.json stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
service = build('sheets', 'v4', credentials=creds)
# Call the Sheets API
sheet = service.spreadsheets()


# Neural network setup

(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
x_tr, x_te = x_tr / 255.0, x_te / 255.0

n = 100

for i in range(n):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128,activation='swish'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        x_tr,
        y_tr,
        epochs=5,
        verbose=0,
        batch_size=30,
        steps_per_epoch=2000
    )

    #print(history.history)

    results = model.evaluate(x_te, y_te, verbose=0)

    #print(results)

    history.history['sparse_categorical_accuracy'].insert(0, results[1])

    values = [
        history.history['sparse_categorical_accuracy']
    ]
    body = {
        'values': values
    }
    result = sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range="Swish!A" + str(i + 2) + ":K" + str(i + 2),
        valueInputOption="RAW",
        body=body
    ).execute()

    tf.keras.backend.clear_session()
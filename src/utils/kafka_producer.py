from time import sleep
from json import dumps
from kafka import KafkaProducer
import pandas as pd
import json
from bson import json_util
import random
import datetime

def run_producer():

    change_df = pd.read_csv('/Users/shahrukh/Documents/CH_INC/data/changes.csv')
    incident_df = pd.read_csv('/Users/shahrukh/Documents/CH_INC/data/incidents.csv')
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                             value_serializer=lambda x:
                             dumps(x).encode('utf-8'), max_request_size=104857600, buffer_memory=104857600)

    incident_df['open_dttm_m'] = pd.to_datetime(incident_df.open_dttm)


    for idx, day_df in incident_df.groupby(incident_df.open_dttm_m.dt.date):

        inc_sub_df = day_df
        min_inc_dt = (datetime.datetime.strptime(inc_sub_df.open_dttm.min(), '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        max_inc_dt = (datetime.datetime.strptime(inc_sub_df.open_dttm.max(), '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        chg_sub_df = change_df[(change_df.act_start_dttm >=min_inc_dt) & (change_df.act_start_dttm <=max_inc_dt)]
        print (len(day_df),min(day_df.open_dttm_m), max(day_df.open_dttm_m))
        print(len(chg_sub_df),chg_sub_df.act_start_dttm.min(),chg_sub_df.act_start_dttm.max())
        print('---------------------')
        inc_data = json.loads(inc_sub_df.to_json(orient='records'))
        chg_data = json.loads(chg_sub_df.to_json(orient='records'))
        if len(chg_sub_df) == 0:
            print('empty change')
            continue
        data = {
        'change_data': chg_data,
        'inc_data': inc_data
        }
        jd = json.dumps(data,ensure_ascii=False)
        producer.send('test4', jd)
        print(len(inc_data),len(chg_data))
        sleep(5)

        # #min_len = min(len(chg_sub_df),50)
        # chg_sub_df = chg_sub_df
        # print(inc_sub_df.open_dttm.min(),inc_sub_df.open_dttm.max())
        #
        # print('period',start)
        # start += 1
        #
        #

        #

        #
        #


if __name__ == '__main__':
    run_producer()

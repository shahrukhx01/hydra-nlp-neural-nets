from kafka import KafkaConsumer
from multiprocessing import Process
import json

# Initializing a queue
queue = []
def run_consumer():
    consumer = KafkaConsumer(
        'test3',
         bootstrap_servers=['localhost:9092'],
         auto_offset_reset='earliest',
         enable_auto_commit=True,
         group_id='my-group',
         value_deserializer=lambda x: json.loads(x.decode('utf-8')));


    for message in consumer:
        message = message.value
        data = json.loads(message)
        queue.append(data)
        print(len(data['inc_data']),len(data['change_data']))

def get_data():
    data_batch = {}
    if len(queue) > 0:
        data_batch = queue.pop(0)
    return data_batch

if __name__ == '__main__':
    run_consumer()

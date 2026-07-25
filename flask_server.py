import os
import sys
import json
import math
import time
import torch
import random
import threading
import traceback
import http.client
import logging
from flask import Flask, request, after_this_request, jsonify
import model
from collections import OrderedDict

# 线程安全的任务存储
running_tasks = OrderedDict()
tasks_lock = threading.Lock()

app = Flask("Standered training process")


def construct_logger(title):
    logger = logging.getLogger(title)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger

# 回调函数封装
def send_callback(api_config, body):
    conn = http.client.HTTPConnection(f"{api_config['ip']}:{api_config['port']}")
    headers = {
        'Content-Type': 'application/json',
        'Content-Length': str(len(body))
    }
    conn.request('POST', api_config['url'], body=body, headers=headers)
    response = conn.getresponse()
    return response.status

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    task_id = data['task_id']

    stop_event = threading.Event()
    stopped_event = threading.Event()

    thread = threading.Thread(target=model.train, args=(data, stop_event, stopped_event,))
    # 存储任务信息
    with tasks_lock:
        running_tasks[task_id] = {
            "thread": thread,
            "stop_event": stop_event,
            "stopped_event": stopped_event,
            "start_time": time.time(),
            "status": "running"
        }
    thread.start()

    return {"code":200, "msg":"Training task successfully started."}


@app.route('/pause', methods=['POST'])
def pause():
    data = request.get_json()
    task_id = data['task_id']
    finish_api = data['finish_api']
    logger = construct_logger(f'Stopping task: {task_id}')
    if task_id in running_tasks:
        running_tasks[task_id]["status"] = "stopping"
        stop_event = running_tasks[task_id]['stop_event']
        stopped_event = running_tasks[task_id]["stopped_event"]

        stop_event.set()
        # 阻塞等待子线程停止确认
        stopped_event.wait()
        running_tasks[task_id]["status"] = "stopped"

        logger.info(f"Task {task_id} is stopped.")
        callback_body = json.dumps({
            'task_id': task_id,
            'code': 200,
            'msg': {
                'task_id': task_id,
                'info': f'Task {task_id} Pause Success.'
            }
        })
    else:
        logger.info(f"Task {task_id} dose not exist.")
        callback_body = json.dumps({
            'task_id': task_id,
            'code': 500,
            'msg': {
                'task_id': task_id,
                'info': f'Task {task_id} dose not exist.'
            }
        })
    send_callback(finish_api, callback_body)
    # 发送回调函数
    return {"code":200, "msg":"Train task successfully stopped."}


@app.route('/resume', methods=['POST'])
def resume():
    print("训练恢复")
    '''
    调用resume脚本恢复训练
    '''
    data = request.get_json()
    task_id = data['task_id']
    data['resume'] = True
    stop_event = threading.Event()
    stopped_event = threading.Event()

    if task_id not in running_tasks:
        logger.info(f"Task {task_id} does not exist.")
        return {'msg': "Training task failed to resume..", 'code': 500}

    else:
        thread = threading.Thread(target=model.train, args=(data, stop_event, stopped_event,))
        # 修改任务信息
        with tasks_lock:
            running_tasks[task_id]["status"] = "retraining"
            running_tasks[task_id]["stop_event"] = stop_event
            running_tasks[task_id]["stopped_event"] = stopped_event
            running_tasks[task_id]["thread"] = thread
        thread.start()
        return {'msg': "Training task successfully resumed.", 'code': 200}


@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    task_id = data['task_id']
    thread = threading.Thread(target=model.infer, args=(data,))
    thread.start()

    return {"code":200, "msg":"Reasoning task successfully started."}


@app.route('/preload', methods=['POST'])
def preload():
    data = request.get_json()
    task_id = data['task_id']

    logger = construct_logger(f'Training task: {task_id}')

    return {"code":200, "msg":"Successfully preloaded."}


@app.route('/freeGPU', methods=['POST'])
def freeGPU():
    data = request.get_json()
    task_id = data['task_id']

    logger = logging.getLogger(f'Predict task: {task_id}')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    
    
    logger.info('Task ID:{0} Start Training'.format(data['task_id']))

    return {"code":200, "msg":"Secceed"}

@app.route('/test', methods=['POST'])
def test():
    data = request.get_json()
    task_id = data['task_id']
    print(data)


# 任务列表端点[2](@ref)
@app.route('/list-tasks', methods=['GET'])
def list_running_tasks():
    """获取当前运行中的任务列表"""
    with tasks_lock:
        tasks = [{
            "task_id": task_id,
            "status": info["status"],
            "start_time": info["start_time"],
            "duration": round(time.time() - info["start_time"], 2)
        } for task_id, info in running_tasks.items()]

    return jsonify({
        "status": "success",
        "tasks": tasks,
        "count": len(tasks)
    }), 200


# 任务状态端点
@app.route('/task-status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取特定任务的详细状态"""
    with tasks_lock:
        if task_id not in running_tasks:
            return jsonify({"status": "error", "message": "Task not found"}), 404

        task_info = running_tasks[task_id]

    return jsonify({
        "status": "success",
        "task_id": task_id,
        "status": task_info["status"],
        "start_time": task_info["start_time"],
        "duration": round(time.time() - task_info["start_time"], 2),
        "alive": task_info["thread"].is_alive()
    }), 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     print("推理开始")
#     '''
#     （1） 解析post消息、上传配置文件、缺省配置文件，组合成算法推理调用参数
#     （2） 调用predict.py(detect.py)，传递推理参数，开始推理
#     '''
#     try:
#         print(request.data)
#         data = request.get_json()
#         sftp_address = data['sftp_address']  # 存储影像的sftp地址
#         sftp_port = data['sftp_port']  # 存储影像的sftp端口
#         sftp_username = data['sftp_username']  # 存储影像的sftp用户名
#         sftp_password = data['sftp_password']  # 存储影像的sftp密码

#         img_txt = data['predict_set_file']
#         log_path = data['predict_log_path']
#         out_path = data['predict_result_path']
#         gpunum = data['predict_set_gpunum']
#         bs = data['predict_set_bs']
#         task_id = data['task_id']
#         predict_model_path = data['predict_model_path']
#         # predict_model_name = data['predict_model_name']

#         sf = paramiko.Transport((sftp_address, sftp_port))
#         sf.connect(username=sftp_username, password=sftp_password)
#         sftp = paramiko.SFTPClient.from_transport(sf)

#         for i in range(10000):
#             if os.path.exists('/usr/local/services/detect/' + str(i)):
#                 continue
#             else:
#                 rootpath = '/usr/local/services/detect/' + str(i) + '/'
#                 os.makedirs(rootpath + 'image/')
#                 imagepath = rootpath + 'image/'
#                 os.makedirs(rootpath + 'model/')
#                 modelpath = rootpath + 'model/'
#                 os.makedirs(rootpath + 'xml/')
#                 xmlpath = rootpath + 'xml/'
#                 os.makedirs(rootpath + 'xml2/')
#                 xmlpath2 = rootpath + 'xml2/'
#                 os.makedirs(rootpath + 'log/')
#                 logpath = rootpath + 'log/'
#                 break

#         sftp.get(img_txt, rootpath + 'testdata.txt')
#         # 将远端服务器下数据下载到'/usr/local/services/detect/get/'
#         fo = open(rootpath + 'testdata.txt', "r")
#         for img_path in fo.readlines():
#             try:
#                 print('downloading' + img_path)
#                 sftp.get(img_path.strip(), imagepath + img_path.strip().split('/')[-1])
#             except:
#                 print("make sure data in predict_set_file(txt) exists")
#         fo.close()

#         # 将远程服务器的pth下载到docker内部
#         print('downloading model')
#         sftp.get(predict_model_path, modelpath + 'predict.pth')

#         f = open(rootpath + 'testdata.txt', "r")
#         # get_path_list = os.listdir('/usr/local/services/detect/get/')
#         for get_path in f.readlines():  # get_path_list:
#             get_path = imagepath + get_path.strip().split('/')[-1]
#             print("img:" + get_path + " is Processing")
#             os.system(
#                 '/root/anaconda3/envs/py37/bin/python /usr/local/services/detect/detect.py --dataset_path ' + get_path + ' --ngpus ' + gpunum + ' --batch_size ' + bs + ' --model_path ' + modelpath + 'predict.pth' + ' --task_id ' + task_id + ' --log_path ' + logpath + ' --outdir ' + xmlpath + ' --outdir2 ' + xmlpath2)

#             outxmlname = xmlpath + os.path.splitext(get_path.split('/')[-1])[0] + '.TargetDetect.result.xml'
#             # outxmlname2 = xmlpath2 + os.path.splitext(get_path.split('/')[-1])[0] + '.report.xml'

#             print(outxmlname)

#             try:
#                 sftp.put(outxmlname, out_path + '/' + outxmlname.split('/')[-1])
#                 # sftp.put(outxmlname2, out_path + '/' + outxmlname2.split('/')[-1])

#                 logname = logpath + os.path.splitext(get_path.split('/')[-1])[0] + '-output.log'
#                 print(logname)
#                 sftp.put(logname, log_path + '/' + logname.split('/')[-1])
#             except:
#                 print('请确认远程服务器log文件夹存在')
#                 # return {'wrong' : '请确认远程服务器log文件夹存在' }
#         f.close()
#         xmllist = os.listdir(xmlpath)
#         xmlstr = ''
#         for xml in xmllist:
#             xmlstr = xmlstr + out_path + '/' + xml + ';'
#         xmlstr = xmlstr[0:-1]

#         xmllist2 = os.listdir(xmlpath2)
#         xmlstr2 = ''
#         for xml2 in xmllist2:
#             xmlstr2 = xmlstr2 + out_path + '/' + xml2 + ';'
#         xmlstr2 = xmlstr2[0:-1]

#         os.system('rm -rf ' + rootpath)
#         return {'msg': "请求成功", 'data': "推理结果信息", 'code': 200, 'predict_result_path': xmlstr}
#     except Exception as e:
#         traceback.print_exc()


# @app.route('/shutdown', methods=['POST'])
# def shutdown():
#     print("停止训练")
#     '''
#     （1） 调用shutdown.py，停止训练
#     '''
#     return {'msg': "请求成功", 'data': "停止结果信息", 'code': 200}


if __name__ == '__main__':
    app.debug = True  # 设置调试模式，生产模式的时候要关掉debug
    app.run(host='0.0.0.0', port=18503, debug=True)

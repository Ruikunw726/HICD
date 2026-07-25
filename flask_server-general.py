import os
import sys
import json
import time
import signal
import traceback
import threading
import http.client
import logging
from flask import Flask, request
# import functions
import subprocess
import psutil
import select

class algorithm_runner():
    '''
    Author     : archie
    Date       : 2025 / 10 / 10
    Description:
        algorithm_runner 类负责将 http 请求转化成可以运行的命令并启动子进程运行算法。
        支持对算法进程进行暂停、继续、杀死（资源释放）以及僵死判断。
        算法需要手动实现训练、推理和评估。
        算法需要提供 config.json 供 algorithm_runner 分析并构建算法启动命令。

        运行逻辑：
        读取config.json --> 解析参数 --> 获取 http 请求 --> 构建算法启动命令 --> 创建子进程启动算法 
                                                                                     |
            杀死进程 <-- 日志超时无输出 <-- 监控算法运行日志 <----------------------------------- 恢复子进程
              |                             |       |                                             |
              |--------------------- 接收结束请求  接收暂停请求 --> 暂停子进程 --> 接收继续执行请求 ---
                                                                       
    Note       : algorithm_runner 一次只能运行一个任务
    '''
    def __init__(self, config_path, timeout=2):
        self.config_file = config_path
        self.task_type = ''
        self.alg_type = ''

        self.task_id = ''
        self.finish_api = {
            'ip': '0.0.0.0',
            'url': '',
            'port': 80
        }
        self.finish_api_pause = {
            'ip': '0.0.0.0',
            'url': '',
            'port': 80
        }
        self.finish_api_unpause = {
            'ip': '0.0.0.0',
            'url': '',
            'port': 80
        }
        self.python_interp = ''
        self.command = ''

        self.command_config = {
            'train':None,
            'infer':None,
            'eval':None,
        }
        self.__config = {}
        self.__config_parser()

        self.__timer = None
        self.__timeout = timeout
        self.__timeout_times = 0
        self.__proc = None
        self.__psutil_proc_handler = None

        self.__logger = None
        self.__logger_handler = None
        self.__proc_logger = None
        self.__proc_logger_handler = None
        self.__construct_logger()

        self.mem_alloc = 1e9
        self.cpu_alloc = 0
        self.gpu_alloc = 0
        self.gpu_mem_alloc = 0


    def __config_parser(self):
        '''
        读取config.json文件，设置启动命令配置
        '''
        with open(self.config_file) as jsonf:
            self.__config = json.load(jsonf)

        self.alg_type = self.__config['info']['algorithmType']

        for elem in self.__config["API/CMD"]:
            request_type = elem['type']
            CPU_requirement = elem['CPU']
            mem_requirement = elem['mem']
            GPU_mem_requirement = elem['GPUMem']
            GPU_num_requirement = elem['GPUNumber']
            
            self.command_config[request_type] = {
                "CPU": CPU_requirement,
                "mem": mem_requirement,
                "GPU_men": GPU_mem_requirement,
                "GPU_num": GPU_num_requirement
            }
            self.command_config[request_type].update(elem['command'])

            self.command_config[request_type]['args'] = {}

        if 'args' in self.__config:
            for elem in self.__config['args']:
                arg_key = elem['argKey']
                for request_type in elem['argAPI']:
                    self.command_config[request_type]['args'][arg_key] = elem
            

    def __task_daemon(self):
        '''
        运行命令，实时监控子进程输出结果，长时间未输出日志则自动终止任务
        '''
        with subprocess.Popen(
            self.command,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid
        ) as self.__proc:
            self.__psutil_proc_handler = psutil.Process(self.__proc.pid)
            # 计时器，监控日志输出时间
            self.__start_timer()

            read_fds = [self.__proc.stdout, self.__proc.stderr]

            self.__proc_print("Task Started")

            while True:
                readable, _, _ = select.select(read_fds, [], [], 0.1)
                
                for fd in readable:
                    line = fd.readline()
                    if line:
                        if fd == self.__proc.stdout:
                            self.__proc_print(line.rstrip())
                        elif self.__contain_string(line, ['error', 'false']):
                            self.__proc_error(line.rstrip())
                        elif self.__contain_string(line, ['warn', 'warning']):
                            self.__proc_warn(line.rstrip())
                        else:
                            self.__proc_error(line.rstrip())

                        self.__cancel_timer()
                        self.__start_timer()

                if self.__proc.poll() is not None:
                    for fd in read_fds:
                        remaining = fd.read()
                        if remaining:
                            if fd == self.__proc.stdout:
                                self.__proc_print(remaining.rstrip())
                            else:
                                self.__proc_error(remaining.rstrip())
                    break

        self.__cancel_timer()
        self.__proc_print("Task Finished")
        self.__clean_proc_logger()

        # 根据命令是否成功调用finish api
        if self.__proc.returncode != 0:
            self.__print(f"task failled, code: {self.__proc.returncode}")
            callback_info = self.__construct_callback_info(500, f"task failled, code: {self.__proc.returncode}")
            self.__send_callback(callback_info)
        else:
            self.__print(f"task success!")
            callback_info = self.__construct_callback_info(200, f"task success!")
            self.__send_callback(callback_info)

    
    def __command_builder(self, request:dict):
        '''
        根据请求内容和启动命令配置，构建启动命令，设置self.command 以及硬件分配指标
        '''
        self.cpu_alloc = self.command_config[self.task_type]['CPU']
        self.mem_alloc = self.command_config[self.task_type]['mem']
        self.gpu_alloc = self.command_config[self.task_type]['GPU_num']
        # pretrain_path = request['pretrain']

        self.command = []

        self.command.append(self.command_config[self.task_type]['prefix'])
        if 'prefixArgs' in self.command_config[self.task_type]:
            self.command += self.command_config[self.task_type]['prefixArgs'].split(' ')
        self.command.append(self.command_config[self.task_type]['runFile'])
        if 'runFileFixedArgs' in self.command_config[self.task_type]:
            self.command += self.command_config[self.task_type]['runFileFixedArgs'].split(' ')
        
        # self.command += [
        #     self.command_config[self.task_type]['prefix'],
        #     self.command_config[self.task_type]['prefixArgs'],
        #     self.command_config[self.task_type]['runFile'],
        #     self.command_config[self.task_type]['runFileFixedArgs']
        # ]

        if self.__key_valid(request, "pretrain"):
            pretrain_path = request['pretrain']
            self.command += [
                self.command_config[self.task_type]['pretrainPathArg'],
                pretrain_path,
            ]

        if self.task_type == 'train':
            training_log = request['training_log']
            input_data = request['input_data']
            output_data = request['output_data']
            self.command += [
                self.command_config[self.task_type]['inputPathArg1'],
                input_data,
                self.command_config[self.task_type]['outputPathArg1'],
                output_data,
                self.command_config[self.task_type]["trainingLogPath"],
                training_log
            ]
            os.makedirs(output_data, exist_ok=True)

        elif self.task_type =='eval':
            input_data = request['input_data']
            output_data = request['output_data']
            self.command += [
                self.command_config[self.task_type]['inputPathArg1'],
                input_data,
                self.command_config[self.task_type]['outputPathArg1'],
                output_data
            ]
            os.makedirs(output_data, exist_ok=True)

        elif self.task_type =='infer':
            # 根据任务类型构建输入数据的命令
            if self.alg_type in ['TDR', 'TSC', 'SRR', 'IT']:
                input_data = next(iter(request['input_data'][0].values()))
                self.command += [
                    self.command_config[self.task_type]['inputPathArg1'],
                    input_data['path']
                ]

                if self.__key_valid(self.command_config[self.task_type], 'inputMetaArg1') and self.__key_valid(input_data, 'meta'):
                    self.command += [
                        self.command_config[self.task_type]['inputMetaArg1'],
                        input_data['meta']
                    ]

                if self.__key_valid(self.command_config[self.task_type], 'sliceArg1') and self.__key_valid(input_data, 'slice'):
                    self.command += [
                        self.command_config[self.task_type]['sliceArg1'],
                        str(input_data['slice'])
                    ]

                if self.__key_valid(self.command_config[self.task_type], 'inputXMLArg1') and self.__key_valid(input_data, 'xml'):
                    self.command += [
                        self.command_config[self.task_type]['inputXMLArg1'],
                        input_data['xml']
                    ]
            elif self.alg_type in ['TCD', 'SCD']:
                input_data = []
                input_xml = []
                digits = []
                for elem in request['input_data'][0]:
                    if elem.isdigit():
                        digits.append(elem)
                if digits[0] > digits[1]:
                    input_data += [request['input_data'][0][digits[1]], request['input_data'][0][digits[0]]]
                else:
                    input_data += [request['input_data'][0][digits[0]], request['input_data'][0][digits[1]]]

                for i in range(2):
                    self.command += [
                        self.command_config[self.task_type][f'inputPathArg{i + 1}'],
                        input_data[i]['path']
                    ]

                    if self.__key_valid(self.command_config[self.task_type], f'inputMetaArg{i + 1}') and self.__key_valid(input_data[i], 'meta'):
                        self.command += [
                            self.command_config[self.task_type][f'inputMetaArg{i + 1}'],
                            input_data[i]['meta']
                        ]

                    if self.__key_valid(self.command_config[self.task_type], f'sliceArg{i + 1}') and self.__key_valid(input_data[i], 'slice'):
                        self.command += [
                            self.command_config[self.task_type][f'sliceArg{i + 1}'],
                            str(input_data[i]['slice'])
                        ]

                    if self.__key_valid(self.command_config[self.task_type], f'inputXMLArg{i + 1}') and self.__key_valid(input_data[i], 'xml'):
                        self.command += [
                            self.command_config[self.task_type][f'inputXMLArg{i + 1}'],
                            input_data[i]['xml']
                        ]
            else:
                self.__error(f'task type {self.task_type} not defined')

            # 根据任务类型设置输出命令，创建文件夹保证文件夹路径一定存在
            if self.alg_type in ['TDR', 'TSC', 'SRR']:
                input_data = next(iter(request['input_data'][0].values()))
                output_path = input_data['output']
                self.command += [
                    self.command_config[self.task_type]['outputPathArg1'],
                    output_path
                ]
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            elif self.alg_type in ['SCD']:
                output_path = request['input_data'][0]['output']
                self.command += [
                    self.command_config[self.task_type]['outputPathArg1'],
                    output_path
                ]
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            elif self.alg_type in ['TCD']:
                output_path = []

                input_data = []
                digits = []
                for elem in request['input_data'][0]:
                    if elem.isdigit():
                        digits.append(elem)
                if digits[0] > digits[1]:
                    input_data += [request['input_data'][0][digits[1]], request['input_data'][0][digits[0]]]
                else:
                    input_data += [request['input_data'][0][digits[0]], request['input_data'][0][digits[1]]]

                output_path.append(input_data[0]['output'])
                output_path.append(input_data[1]['output'])
                output_path.append(request['input_data'][0]['output'])
                for i in range(3):
                    if f'outputPathArg{i + 1}' not in self.command_config[self.task_type]: continue
                    if not self.command_config[self.task_type][f'outputPathArg{i + 1}']: continue
                    self.command += [
                        self.command_config[self.task_type][f'outputPathArg{i + 1}'],
                        output_path[i]
                    ]
                    os.makedirs(os.path.dirname(output_path[i]), exist_ok=True)

            elif self.alg_type in ['IT']:
                input_data = next(iter(request['input_data'][0].values()))
                output_path = []
                output_path.append(input_data['output'])
                output_path.append(input_data['json_output'])
                print(output_path)
                for i in range(2):
                    print(self.command_config[self.task_type][f'outputPathArg{i + 1}'])
                    self.command += [
                        self.command_config[self.task_type][f'outputPathArg{i + 1}'],
                        output_path[i]
                    ]
                    os.makedirs(os.path.dirname(output_path[i]), exist_ok=True)

        # 设置必须超参数的值
        args_set = self.command_config[self.task_type]['args']
        necessary_args = {}
        for arg in args_set:
            if args_set[arg]["argRequired"]:
                necessary_args[arg] = (
                    args_set[arg]["argFlags"][0],
                    args_set[arg]["argDefault"]
                )

        # 根据发送的请求修改超参数默认值，并将非必须超参数添加进去
        if "parameters" in request:
            for arg in request['parameters']:
                necessary_args[arg] = (
                    args_set[arg]["argFlags"][0],
                    request['parameters'][arg]
                )

        for arg_flag, value in necessary_args.values():
            self.command += [arg_flag, str(value)]

        # self.command = map(lambda x: '\"' + x + '\"' if ' ' in x else x, self.command)

        while '' in self.command:
            self.command.remove('')

        # self.command = 'python -u test_subprocess.py'

    # todo
    def train(self, request):
        self.__print('Train request')

        data = request.get_json()
        self.finish_api = data['finish_api']
        self.task_type = 'train'
        self.task_id = data['task_id']
        try:
            self.__command_builder(data)
        except Exception as e:
            self.__error(str(e))
        self.__print("RUN:" + ' '.join(self.command))
        # print(f"command:{self.command}")
        self.__construct_proc_logger()
        thread = threading.Thread(target=self.__task_daemon)
        thread.start()
        
    # todo
    def infer(self, request):
        self.__print('Infer request')

        data = request.get_json()
        self.finish_api = data['finish_api']
        self.task_type = 'infer'
        self.task_id = data['task_id']
        try:
            self.__command_builder(data)
        except Exception as e:
            self.__error(str(e))
            tb_str = traceback.format_exc()
            self.__error(tb_str)
        self.__print("RUN:" + ' '.join(self.command))
        # print(f"command:{self.command}")
        self.__construct_proc_logger()
        thread = threading.Thread(target=self.__task_daemon)
        thread.start()

    # todo
    def eval(self, request):
        self.__print('Eval request')

        data = request.get_json()
        self.finish_api = data['finish_api']
        self.task_type = 'eval'
        self.task_id = data['task_id']
        try:
            self.__command_builder(data)
        except Exception as e:
            self.__error(str(e))
        self.__print("RUN:" + ' '.join(self.command))
        # print(f"command:{self.command}")
        self.__construct_proc_logger()
        thread = threading.Thread(target=self.__task_daemon)
        thread.start()


    def pause(self, request):
        self.__print('Pause request')
        data = request.get_json()
        self.finish_api_pause = data['finish_api']
        try:
            
            if self.__proc is None or self.__proc.poll() is not None:
                raise Exception("Task not running.")
            self.__psutil_proc_handler.suspend()
            self.__cancel_timer()
            callback_info = self.__construct_callback_info(200, 'Pause success.')
            self.__send_callback(callback_info, 'pause')
            self.__print('200, Pause success.')
        except Exception as e:
            callback_info = self.__construct_callback_info(500, 'Pause failed.\n' + str(e))
            self.__send_callback(callback_info, 'pause')
            self.__error('500, Pause failed.\n' + str(e))


    def unpause(self, request):
        self.__print('Task continue request')
        data = request.get_json()
        self.finish_api_unpause = data['finish_api']
        try:
            # self.__psutil_proc_handler.resume()
            if self.__proc is None or self.__proc.poll() is not None:
                raise Exception("Task not paused.")
            self.__psutil_proc_handler.resume()
            self.__start_timer()
            callback_info = self.__construct_callback_info(200, 'Task continue success.')
            self.__send_callback(callback_info, 'unpause')
            self.__print('200, Task continue success.')
        except Exception as e:
            callback_info = self.__construct_callback_info(500, 'Task continue failed.\n' + str(e))
            self.__send_callback(callback_info, 'unpause')
            self.__error('500, Task continue failed.')
            self.__error(str(e))


    def stop(self, request):
        self.__print('Task stop request')
        # stop use the finish api of infer, eval and train
        # data = request.get_json()
        # self.finish_api = data['finish_api']
        try:
            self.__kill()
            self.__cancel_timer()
            callback_info = self.__construct_callback_info(200, 'Task stop success!')
            self.__send_callback(callback_info)
            self.__print('200, Task stop success.')
        except Exception as e:
            callback_info = self.__construct_callback_info(500, 'Task stop failed.\n' + str(e))
            self.__send_callback(callback_info)
            self.__error('500, Task stop failed.\n')
            self.__error(str(e))


    def __construct_logger(self):
        self.__logger = logging.getLogger('Flask Server')
        self.__logger.setLevel(logging.DEBUG)
        self.__logger_handler = logging.StreamHandler(sys.stdout)
        self.__logger_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        self.__logger_handler.setFormatter(formatter)
        self.__logger.addHandler(self.__logger_handler)


    def __construct_proc_logger(self):
        self.__proc_logger = logging.getLogger(self.task_type + '-' + self.task_id)
        self.__proc_logger.setLevel(logging.DEBUG)
        self.__proc_logger_handler = logging.StreamHandler(sys.stdout)
        self.__proc_logger_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        self.__proc_logger_handler.setFormatter(formatter)
        self.__proc_logger.addHandler(self.__proc_logger_handler)


    def __clean_proc_logger(self):
        for handler in self.__proc_logger.handlers[:]:
            handler.close()
            self.__proc_logger.removeHandler(handler)
        # logging.Manager().removeLogger(self.__proc_logger)


    def __proc_print(self, info):
        self.__proc_logger.info(info)


    def __proc_error(self, error):
        self.__proc_logger.error(error)


    def __proc_warn(self, warning):
        self.__proc_logger.warning(warning)


    def __print(self, info):
        self.__logger.info(info)


    def __error(self, error):
        self.__logger.error(error)


    def __time_out(self):
        '''
        子进程僵死判断，若内存接近分配限额，或cpu占用接近0，则判断僵死，
        否则判断仍在运行，继续等待一轮，若下一轮仍未输出内容，则直接杀死。
        '''
        if self.__proc is None or self.__proc.poll() is not None:
            self.__error(f"Timer triggered but task already exited.")
            self.__timeout_times = 0
            return

        self.__error(f"Task output time out.")
        self.__timeout_times += 1
        mem_alloc_MB = self.__psutil_proc_handler.memory_info().rss / 1024 / 1024
        cpu_percent = self.__psutil_proc_handler.cpu_percent(interval=0.1)
        if mem_alloc_MB / self.mem_alloc > 0.98:
            self.__error('Memory reached maximum, task is probablly dead')
            self.__kill()
        elif cpu_percent < 0.1:
            self.__error('CPU occupation too low, task is probablly dead')
            self.__kill()
        elif self.__timeout_times < 1:
            self.__print('Task is probablly still running, reset time clock.')
            self.__cancel_timer()
            self.__start_timer()
        else:
            self.__error('Task time out, probablly dead')
            self.__kill()


    def __kill(self):
        if self.__proc is None or self.__proc.poll() is not None:
            self.__error(f"Task already terminated or not running.")
            return
        try:
            pgid = os.getpgid(self.__proc.pid)
            self.__error(f"Killing process group (PGID: {pgid}) including all child processes.")

            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            if self.__proc.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
                self.__error(f"Force killed process group (PGID: {pgid}) with SIGKILL.")

            if self.__psutil_proc_handler:
                child_processes = self.__psutil_proc_handler.children(recursive=True)
                for child in child_processes:
                    try:
                        if child.is_running():
                            child.terminate()
                            time.sleep(0.5)
                            if child.is_running():
                                child.kill()
                            self.__error(f"Killed child process (PID: {child.pid}).")
                    except psutil.NoSuchProcess:
                        continue
                    except Exception as e:
                        self.__error(f"Failed to kill child process (PID: {child.pid}): {str(e)}")

            self.__proc.wait(timeout=3)
            self.__error(f"Main task process (PID: {self.__proc.pid}) terminated successfully.")

        except OSError as e:
            self.__error(f"Failed to kill process group: {str(e)}. Trying fallback kill.")
            self.__proc.terminate()
            time.sleep(1)
            if self.__proc.poll() is None:
                self.__proc.kill()
        finally:
            # 重置状态
            self.__proc = None
            self.__psutil_proc_handler = None
            self.__timeout_times = 0


    def __start_timer(self):
        '''
        创建定时器，超时未取消则自动杀死进程
        '''    
        self.__timer = threading.Timer(self.__timeout, self.__time_out)
        self.__timer.start()


    def __cancel_timer(self):
        '''
        取消定时器，超时未取消则自动杀死进程
        '''
        self.__timer.cancel()


    def __construct_callback_info(self, code:int, info:str):
        return json.dumps({
                    'task_id': self.task_id,
                    'code': code,
                    'msg': info
                })


    def __send_callback(self, body, mode:str = ''):
        if mode == 'pause':
            finish_api = self.finish_api_pause
        elif mode == 'unpause':
            finish_api = self.finish_api_unpause
        else:
            finish_api = self.finish_api

        conn = http.client.HTTPConnection(f"{finish_api['ip']}:{finish_api['port']}")
        headers = {
            'Content-Type': 'application/json',
            'Content-Length': str(len(body))
        }
        conn.request('POST', finish_api['url'], body=body, headers=headers)
        response = conn.getresponse()

    
    def __key_valid(self, x:dict, key):
        return key in x and x[key]
    

    def __contain_string(self, str1:str, strlist:list) -> bool:
        for str_ in strlist:
            if str_.lower() in str1.lower():
                return True
            
        return False
        

if __name__ == '__main__':
    app = Flask("Algorithm flask server")
    x = algorithm_runner('./config.json', timeout=600)

    @app.route('/train', methods=['POST'])
    def train():
        try:
            x.train(request)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            return {"code":500, "msg":"Error:" + str(e)}
        return {"code":200, "msg":"Training task request received."}
    
    @app.route('/infer', methods=['POST'])
    def infer():
        try:
            x.infer(request)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
        return {"code":200, "msg":"Reasoning task request received."}
    
    @app.route('/eval', methods=['POST'])
    def eval():
        try:
            x.eval(request)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            return {"code":500, "msg":"Error:" + str(e)}
        return {"code":200, "msg":"Eval task request received."}
    
    @app.route('/pause', methods=['POST'])
    def pause():
        try:
            x.pause(request)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            return {"code":500, "msg":"Error:" + str(e)}
        return {"code":200, "msg":"Task paused."}

    @app.route('/unpause', methods=['POST'])
    def unpause():
        try:
            x.unpause(request)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            return {"code":500, "msg":"Error:" + str(e)}
        return {"code":200, "msg":"Task continued request received."}
    
    @app.route('/stop', methods=['POST'])
    def stop():
        try:
            x.stop(request)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            return {"code":500, "msg":"Error:" + str(e)}
        return {"code":200, "msg":"Task stop request received."}

    # 关闭flask自带日志输出
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=18888, debug=False)

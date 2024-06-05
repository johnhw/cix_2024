import zmq, json, sys
from multiprocessing import Process, Queue
import queue
import subprocess
import time

def json_encode(obj):
    return json.dumps(obj).encode("utf-8")

def json_decode(obj):
    return json.loads(obj.decode("utf-8"))

def async_req_loop(zmq_port, in_q, out_q):
    # Create an infinite loop that connects to a ZMQ port,
    # sends a request (from in_q), and waits for a response
    # (which is placed in out_q).
    # Messages are JSON encoded.
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{zmq_port}")
    while True:  
        try:
            request = in_q.get()
            #print("Got request")
            # get *all* waiting requests from the queue
            while not in_q.empty():
                request = in_q.get_nowait()
        except queue.Empty:
            pass
        
        socket.send(json_encode(request))
        #print("Send request")
        response = socket.recv()
        #print("Received response")
        try:
            out_q.put_nowait(json_decode(response))
            #print("Put response")
        except queue.Full:
            pass


def sync_rep_loop(zmq_port, fn, timeout=2000):
    # synchronous reply loop
    # Create an infinite loop that binds to a ZMQ port,
    # waits for a request, and sends a response using
    # the callback fn to process the request.
    # Messages are JSON encoded.
    context = zmq.Context() 
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{zmq_port}")
    socket.setsockopt(zmq.RCVTIMEO, timeout)
    try:
        while True:
            try:
                message = socket.recv()
                response = fn(json_decode(message))
                socket.send(json_encode(response))
            except zmq.error.Again:
                pass
                #print("Timed out")
    except Exception as e:
        print(e)
    finally:
        socket.close()

def rep_loop_test(zmq_port):
    def print_respond(x):
        print(x)
        return {"response": "OK"}
    sync_rep_loop(zmq_port, print_respond)

def req_loop_feed(zmq_port, in_q, out_q):    
    # feed the req loop with some test data
    for i in range(10):
        in_q.put({"test": i})
        print(out_q.get())

def loop_test(zmq_port):
    in_q = Queue()
    out_q = Queue()
    # now the req loop is also in its own process
    process = Process(target=async_req_loop, args=(zmq_port, in_q, out_q))
    process.daemon = True
    process.start()

    # start the req loop tester in its own process
    process = Process(target=req_loop_feed, args=(zmq_port, in_q, out_q))
    process.daemon = True
    process.start()
    # now engage synchronous rep loop
    rep_loop_test(zmq_port)
    
def pub_loop(self, port, topic, pub_q):
    # take messages off pub_q as they arrive,
    # and publish on the given port/topic
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    while True:
        message = pub_q.get()
        socket.send_multipart([topic, json_encode(message)])

def sub_loop(self, port, topic, sub_q):
    # subscribe to the given port/topic, and place
    # messages on sub_q as they arrive
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    while True:
        topic, message = socket.recv_multipart()
        sub_q.put(json_decode(message))
          



def safe_launch(script_path, args=[], timeout=60, sudo=False):
    cmd = [sys.executable, script_path]
    if sudo:
        cmd = ['sudo'] + cmd
    process = subprocess.Popen(cmd + args, creationflags=subprocess.CREATE_NEW_CONSOLE)
    if timeout==0: # no block mode
        return
    time.sleep(timeout)
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()


def start_test():
    # create a process for each of req_rep_loop and rep_loop_test
    # create queues, and then start the processes
    zmq_port = 5555
    loop_test(zmq_port)
    

if __name__=="__main__":
    start_test()

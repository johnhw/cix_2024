import zmq, json, sys
from multiprocessing import Process, Queue

import time

def json_encode(obj):
    return json.dumps(obj).encode("utf-8")

def json_decode(obj):
    return json.loads(obj.decode("utf-8"))

def req_rep_loop(zmq_port, in_q, out_q):
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
            while not in_q.empty():
                request = in_q.get_nowait()
        except queue.Empty:
            pass
        socket.send(json_encode(request))
        response = socket.recv()
        out_q.put(json_decode(response))

def rep_loop(zmq_port, fn):
    # Create an infinite loop that binds to a ZMQ port,
    # waits for a request, and sends a response.
    # Messages are JSON encoded.
    context = zmq.Context() 
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{zmq_port}")
    while True:
        message = socket.recv()
        response = fn(message)
        socket.send(json_encode(response))


def rep_loop_test(zmq_port):
    def print_respond(x):
        print(x)
        return {"response": "OK"}
    rep_loop(zmq_port, print_respond)

def req_loop_test(zmq_port):
    in_q = Queue()
    out_q = Queue()
    process = Process(target=req_rep_loop, args=(zmq_port, in_q, out_q))
    process.daemon = True
    process.start()
    for i in range(10):
        in_q.put({"test": i})
        print(out_q.get())
    
def start_test():
    # create a process for each of req_rep_loop and rep_loop_test
    # create queues, and then start the processes
    p = Process(target=rep_loop_test, args=(5555,))
    p.daemon = True
    p.start()
    
    req_loop_test(5555)
    
    time.sleep(2)
    
    

if __name__=="__main__":
    start_test()

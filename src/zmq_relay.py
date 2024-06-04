import time
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
import zmq


def zmq_server(out_q, ctrl_q, topic="demo", ip="127.0.0.1", port=5556):
    stopped = False
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{ip}:{port}")
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf8"))
    
    while True:        
        if not ctrl_q.empty():
            msg = ctrl_q.get_nowait()
            if msg == "stop":
                print("STOPPING")
                break        
        topic, msg = sock.recv_multipart()        
        out_q.put(msg)


class Relay:
    """Relay ZMQ packets to the host, always
    returning new packets by running the poll
    loop in a separate process"""

    def __init__(self, ip="127.0.0.1", port=5556):
        self.port = port
        self.ip = ip
        self.last_live = datetime.now() - timedelta(days=1)
        self.in_q = Queue()
        self.ctrl_q = Queue()
        self.address = f"{self.ip}:{self.port}"
        self.process = Process(
            target=zmq_server,
            args=(self.in_q, self.ctrl_q),
            kwargs={"port": port, "ip": ip},
        )
        self.process.start()

    def poll(self):
        if self.in_q.empty():
            return None
        self.last_live = datetime.now()
        msg = self.in_q.get()        
        return msg

    def live(self):
        """Return True if the data is fresh (i.e.
        poll successful within 5 seconds)"""
        return (datetime.now() - self.last_live).total_seconds() < 5.0

    def close(self):
        self.ctrl_q.put("stop")
        time.sleep(0.5)
        self.process.terminate()


if __name__ == "__main__":
    import json

    topic = "demo".encode("utf8")
    address = "tcp://127.0.0.1:5556"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    print(f"Opening ZMQ pub on {address}")
    sock.bind(address)
    time.sleep(0.1)

    i = 0
    while True:
        sock.send_multipart([topic, json.dumps({"message_id": i}).encode("utf8")])
        print(topic, i)
        i += 1
        time.sleep(0.5)

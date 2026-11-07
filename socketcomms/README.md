# Socket communications

The transport under RL Spin Decoupler. It needs nothing beyond the standard library and
knows nothing about reinforcement learning, so it can be used on its own for any
request and reply exchange between two Python processes.

There are two kinds of point, in `socketcomms/comms.py`:

- `ServerCommPoint(port)` binds the port on every interface and accepts one client
  when `begin(timeoutaccept)` is called.
- `ClientCommPoint(ip, port)` connects to a server when `begin()` is called.

Both derive from `BaseCommPoint`, which provides `sendData()`, `readData()` and
`checkDataToRead()`. A small example:

```python
# process 1
from spindecoupler import ServerCommPoint

server = ServerCommPoint(49054)
server.begin(timeoutaccept=60.0)
error, data = server.readData(timeout=5.0)
server.sendData({"echo": data})
```

```python
# process 2
from spindecoupler import ClientCommPoint

client = ClientCommPoint("127.0.0.1", 49054)
client.begin()
client.sendData({"hello": 1})
error, answer = client.readData(timeout=5.0)
```

## Protocol

Each message is one frame: the length of the payload in 4 bytes (big endian) followed by
the payload, a pickled dictionary. TCP is a byte stream, so the length is what lets the
receiver know where a message ends, and `readData()` reads exactly that many bytes
before unpickling.

Both ends set `TCP_NODELAY`. A step of the decoupler is a short request followed by two
short replies, and without the option the second reply waits for the acknowledgement of
the first, which the receiver delays by up to 40 ms on Linux. On loopback this exchange
took 44 ms without the option and 0.14 ms with it.

## Errors

Failures of the connection do not raise. `sendData()` returns the error text, and
`readData()` returns a pair `(error, data)`, with an empty `error` on success. The
timeout of `readData()` applies to each socket read, and the error text is `timed out`
when the other side stays silent. Misuse, such as sending through a point that has not
begun, does raise `RuntimeError`.

## Limits

- Ports must be between 20000 and 49151.
- A server accepts one client. Calling `begin()` again closes the current connection
  and waits for a new one.
- If the port is busy, the server constructor retries every 13 seconds, up to 10 times.
- Messages are pickled, and unpickling data from a peer can run arbitrary code. Use the
  module only between processes that trust each other.

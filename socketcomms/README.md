# Socket communications

The transport under RL Spin Decoupler. It needs nothing beyond the standard library and
knows nothing about reinforcement learning, so it can be used on its own for any
request and reply exchange between two Python processes.

There are two kinds of point, in `socketcomms/comms.py`:

- `ServerCommPoint(port)` binds the port on every interface (with `SO_REUSEADDR` on
  POSIX, so a restarted server does not wait for `TIME_WAIT`) and accepts one client
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

Each message is one frame: a 7-byte header followed by the payload, a pickled
dictionary. The header holds the magic `SD`, the protocol version (1) and the length of
the payload in 4 bytes (big endian). TCP is a byte stream, so the length is what lets the
receiver know where a message ends, and `readData()` reads exactly that many bytes
before unpickling. The magic and the version are checked on every frame, so a peer that
is not this protocol, or comes from an incompatible release, gets a clear error instead
of a pickle failure.

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

## Safety options

All three are arguments of `ServerCommPoint` and `ClientCommPoint` (and, as keywords, of
`RLSide` and `AgentSide`).

- `max_frame_size` (64 MiB by default) bounds the payload that is sent or accepted. The
  length comes from the peer, so without it a corrupt header would make the receiver
  try to buffer up to 4 GiB. An incoming frame over the limit, or with a wrong header,
  closes the connection, because the rest of the stream cannot be interpreted.
- `allow_pickle` (`False` by default). Incoming payloads are unpickled with an
  allowlist: plain data (numbers, strings, bytes, lists, tuples, dicts, sets), NumPy
  arrays and scalars, and a few builtins. Any other class, such as `os.system`, is
  refused with an error. Set it to `True` only between endpoints you trust, to send
  your own classes: a general pickle can run arbitrary code.
- `send_timeout` (10 s by default). `sendall` blocks for good once the peer stops
  reading and the socket buffers fill up. After the timeout the send fails and the
  connection is closed, because part of the frame may have been written. 0 or less
  waits without limit.

## Logging

Messages go to the `spindecoupler` logger (`logging`), not to `print`. `verbose=True`
on `RLSide` and `AgentSide` shows them on the standard output; an application can also
configure that logger itself. `setDebug()` logs each send, receive and peek at debug
level.

## Limits

- Ports must be between 20000 and 49151.
- A server accepts one client. Calling `begin()` again closes the current connection
  and waits for a new one.
- If the port is busy, the server constructor retries every 13 seconds, up to 10 times.
- Messages are pickled. The default allowlist blocks the usual attacks, but pickle is
  not a hardened format: use the module between endpoints you trust, and never expose
  the port to an untrusted network.

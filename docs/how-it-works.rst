How It Works
============

The pattern
-----------

The learner and the agent run as separate processes and talk through one TCP
connection. This helps when the agent must run a real-time or nearly real-time control
loop, when the RL library expects a synchronous ``reset()`` and ``step()``, and when the
two cannot share a thread or a notion of time.

Commands
--------

The RL side sends one of three commands:

``reset``
   The agent resets its episode and answers with the first observation.

``step``
   The agent receives a new action, answers with ``LAT`` (the duration of the previous
   action) and then with the observation that follows the new action, plus a reward if it
   computes one.

``finish``
   The agent leaves its loop.

Clocks
------

Two clocks are kept apart on purpose. The agent's clock gives ``LAT`` and ``ATO`` (the
time at which the observation was sampled). The RL process can add ``t_wall``, read with
``time.time()`` when the answer arrives. The values are related but not
interchangeable: ``ATO`` and ``t_wall`` differ by the network delay, the serialization
time, any queueing, and the offset between the two hosts' clocks when they are not
synchronized.

Polling
-------

The agent calls :meth:`spindecoupler.AgentSide.readWhatToDo` on every iteration of its
loop. The method first checks whether the socket has data; if not it returns ``None`` at
once, so the loop keeps running while the learner thinks.

Transport
---------

Payloads are pickled and sent over TCP, each preceded by a 7-byte header: the magic
``SD``, the protocol version and the payload length. TCP is a byte stream, so a single
``send`` on one side does not correspond to a single ``recv`` on the other, and the
length tells the receiver where a message ends. The magic and the version are checked on
every frame, so mismatched releases fail with a clear message.

The length comes from the peer, so ``max_frame_size`` (64 MiB by default) bounds it; a
frame over the limit closes the connection. A send that blocks because the peer stopped
reading gives up after ``send_timeout`` (10 s by default).

Both ends set ``TCP_NODELAY``. A step is a short request followed by two short replies
(``LAT`` and the observation). With Nagle's algorithm on, the second reply waits for the
acknowledgement of the first, and the receiver delays that acknowledgement by up to
40 ms on Linux. On loopback the exchange took 44 ms without the option and 0.14 ms with
it.

Unpickling arbitrary data can run arbitrary code. By default the receiver only accepts
plain data and NumPy arrays and scalars, and refuses any other class; pass
``allow_pickle=True`` to ``RLSide`` and ``AgentSide`` to send your own classes between
endpoints you trust.

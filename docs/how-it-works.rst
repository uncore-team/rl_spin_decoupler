How It Works
============

Decoupling pattern
------------------

The package implements a simple but useful pattern: the learning loop and the
agent loop live in separate processes and synchronize through a TCP connection.

This is helpful when:

- the agent must run a real-time or near-real-time control loop,
- the RL library expects a synchronous ``reset()/step()`` API,
- and the two loops cannot safely share a single thread or a single timing
  model.

Command protocol
----------------

The RL side sends one of three command kinds:

- ``reset``: reset the agent-side episode and return the first observation.
- ``step``: apply a new action, return ``LAT`` for the previous action, then
  return the new observation and optional reward.
- ``finish``: stop the experiment cleanly.

Clock domains
-------------

The design intentionally preserves two time domains.

Agent clock
~~~~~~~~~~~

The agent loop owns:

- ``LAT``: duration of the previous action,
- ``ATO``: timestamp of observation acquisition.

RL clock
~~~~~~~~

The RL loop can add:

- ``t_wall``: local wall-clock time when the RL process receives a response.

These values are related but not interchangeable. In particular, ``ATO`` and
``t_wall`` may differ because of network latency, serialization overhead, queue
delays, or the fact that the processes run on different hosts.

Query channel
-------------

Besides the ``reset()/step()`` cycle, which is driven by the RL side, the
package offers an optional query channel driven by the agent side. It is meant
for deployment, once a policy has been trained: the agent asks for the action
that the policy would take for a given observation, without advancing any
episode.

- :class:`spindecoupler.RLSideQuery` is the server. It waits for a query with
  :meth:`~spindecoupler.RLSideQuery.receive_query`, computes the action, and
  answers with :meth:`~spindecoupler.RLSideQuery.send_action`. If the agent
  disconnects, :meth:`~spindecoupler.RLSideQuery.reconnect` waits for a new
  one on the same port.
- :class:`spindecoupler.AgentSideQuery` is the client. Each call to
  :meth:`~spindecoupler.AgentSideQuery.query_action` sends one observation and
  blocks until the action arrives.

The observation and action formats are not fixed by the library, so a single
query can carry a batch of observations (for example, a list of lists) as long
as both sides agree on it. The query channel uses its own TCP connection, so it
needs its own port if it runs alongside an :class:`~spindecoupler.RLSide`
channel.

Non-blocking polling
--------------------

The agent loop is allowed to keep running even when no RL command is pending.
The method :meth:`spindecoupler.AgentSide.readWhatToDo` first checks whether the
socket has data available; if not, it returns ``None`` immediately.

Transport model
---------------

Payloads are serialized with Python ``pickle`` and sent over TCP with explicit
message framing. This framing is important because TCP is a byte stream: a
single ``send`` on one side is not guaranteed to match a single ``recv`` on the
other side without an application-level frame boundary.

Both endpoints disable Nagle's algorithm (``TCP_NODELAY``). During a step the
agent sends two messages in a row without any reply in between (``LAT`` and,
once the action has been executed, the observation). With Nagle's algorithm
enabled, the second one could be held until the delayed ACK of the first one
arrives, which adds latency spikes of tens of milliseconds whenever actions are
shorter than that in wall-clock time.
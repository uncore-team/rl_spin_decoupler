# SPDX-License-Identifier: GPL-3.0-only

"""
Socket transport for rl_spin_decoupler.

A server point and a client point exchange pickled dictionaries over one TCP
connection. Each message is a frame: a 7-byte header (the magic ``SD``, the
protocol version and the payload length, big endian) followed by the pickled
payload. Failures are reported by returning the error text (an empty string means
success) rather than by raising, except for misuse such as sending through a
point that has not begun.

Received payloads are unpickled with an allowlist of types by default, since
unpickling arbitrary data runs arbitrary code (see ``allow_pickle``).

(c) Juan-Antonio Fernández-Madrigal, 2025
https://babel.isa.uma.es/jafma
"""

import datetime
import errno
import io
import ipaddress
import logging
import os
import pickle
import select
import socket
import struct
import sys
import time
from enum import Enum
from typing import Dict, Tuple

# Version of the frame layout. Both ends must agree; it is checked on every frame so
# that a mismatched release fails with a clear message instead of a pickle error.
PROTOCOL_VERSION = 1

# Largest payload accepted or sent by default. The length in the header comes from
# the peer, so without a limit a corrupt or hostile header makes the receiver try
# to buffer up to 4 GiB.
DEFAULT_MAX_FRAME_SIZE = 64 * 1024 * 1024

_MAGIC = b"SD"
_HEADER = struct.Struct("!2sBI")  # magic, protocol version, payload length

logger = logging.getLogger("spindecoupler")
logger.addHandler(logging.NullHandler())


class _StdoutHandler(logging.StreamHandler):
    """Write to the current ``sys.stdout``, looked up at every emit."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def enable_console_logging(level: int = logging.INFO):
    """
    Show the library's log messages on the standard output.

    The ``verbose`` option of the endpoints calls this. It is idempotent: the
    handler is added once, and only the level can change. An application that
    routes the ``spindecoupler`` logger itself can leave ``verbose`` off and set the
    level of that logger instead.
    """
    logger.setLevel(
        level if logger.level == logging.NOTSET else min(level, logger.level)
    )
    if not any(isinstance(h, _StdoutHandler) for h in logger.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


class _FrameError(RuntimeError):
    """A frame that breaks the protocol; the stream cannot be trusted afterwards."""


class _RestrictedUnpickler(pickle.Unpickler):
    """
    Unpickler that only resolves a short list of harmless globals.

    Plain data (numbers, strings, bytes, lists, tuples, dicts, sets) needs no
    global at all. The list adds what typical observations and actions use:
    NumPy arrays and scalars, a few builtins and ``OrderedDict``. Anything else,
    such as ``os.system``, is refused.
    """

    _ALLOWED = {
        ("builtins", "complex"),
        ("builtins", "bytearray"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "slice"),
        ("builtins", "range"),
        ("collections", "OrderedDict"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
    }
    # NumPy moved these modules between 1.x (core) and 2.x (_core).
    _NUMPY_MODULES = {
        "numpy.core.multiarray": {"_reconstruct", "scalar"},
        "numpy._core.multiarray": {"_reconstruct", "scalar"},
        "numpy.core.numeric": {"_frombuffer"},
        "numpy._core.numeric": {"_frombuffer"},
    }

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED or name in self._NUMPY_MODULES.get(
            module, ()
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            "refused to unpickle {}.{}; build the point with allow_pickle=True "
            "only if you trust the peer".format(module, name)
        )


class BaseCommPoint:
    """
    Behaviour shared by the server and the client point: argument validation,
    framing, and the send and receive primitives.
    """

    class Kind(Enum):
        """Role of a point in the connection."""

        SERVER = 0
        CLIENT = 1

    @classmethod
    def get_ip(cls):
        """
        Return the IPv4 address of the interface used to reach other networks, or
        ``127.0.0.1`` if there is none.

        Connecting a UDP socket only selects a route, so nothing is sent and the
        address used below does not have to be reachable.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("10.254.254.254", 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = "127.0.0.1"
        finally:
            s.close()
        return IP

    def __init__(
        self,
        kind: Kind,
        port: int = 49054,
        ipv4: str = "127.0.0.1",
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
        allow_pickle: bool = False,
        send_timeout: float = 10.0,
    ):
        """
        Args:
            kind: Role of the point.
            port: TCP port, between 20000 and 49151.
            ipv4: IPv4 address the point refers to: the address to bind for a
                server, the address to connect to for a client.
            max_frame_size: Largest payload, in bytes, that the point sends or
                accepts. A larger incoming frame closes the connection, because
                the stream cannot be resynchronized.
            allow_pickle: If False (default), incoming payloads are unpickled with
                an allowlist of types (plain data, NumPy arrays and scalars). If
                True, any pickle is accepted, which lets the peer run arbitrary
                code; use it only between endpoints you trust.
            send_timeout: Seconds a send may block when the peer stops reading. 0
                or less waits without limit. A send that times out closes the
                connection, since part of the frame may have been written.

        Raises:
            TypeError: If ``kind`` is not a :class:`Kind`.
            ValueError: If any other argument is invalid.
        """
        if not isinstance(kind, BaseCommPoint.Kind):
            raise (TypeError("Expected a Kind argument, got {}".format(type(kind))))
        if (
            (not isinstance(max_frame_size, int))
            or isinstance(max_frame_size, bool)
            or not (0 < max_frame_size <= 0xFFFFFFFF)
        ):
            raise (
                ValueError(
                    "The max. frame size {} is invalid; it should be an integer "
                    "between 1 and 4294967295".format(max_frame_size)
                )
            )
        if not isinstance(send_timeout, (int, float)) or isinstance(send_timeout, bool):
            raise (ValueError("The send timeout {} is invalid".format(send_timeout)))
        if (not isinstance(port, int)) or not (20000 <= port <= 49151):
            raise (
                ValueError(
                    "Port {} is invalid; it should be an integer between "
                    "20000 and 49151".format(port)
                )
            )
        try:
            ipaddress.IPv4Address(ipv4)
        except ipaddress.AddressValueError:
            raise (ValueError("IP address {} is invalid".format(ipv4)))

        self._kind = kind
        self._max_frame_size = max_frame_size
        self._allow_pickle = allow_pickle
        self._send_timeout = send_timeout
        self._port = port
        self._ipv4 = ipv4
        self._begun = False  # set by begin() in the derived classes
        self._debug = False

    @staticmethod
    def _disable_nagle(sock: socket.socket):
        """
        Send each message as soon as it is written (set TCP_NODELAY).

        Without this, Nagle's algorithm holds back a small write while an earlier
        one is still unacknowledged. A protocol step is a short request followed
        by two short replies sent back to back (duration of the previous action,
        then the observation), so the second reply waited for the ACK of the
        first, and the receiver delays that ACK (up to 40 ms on Linux). Every step
        paid that wait and the LAT reported by the agent included it. On loopback
        this exchange took 44 ms before the change and 0.14 ms after. The
        messages are tiny, so coalescing them saves nothing.

        Set it on both ends: the socket returned by ``accept()`` does not inherit
        the option from the listening socket.
        """
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def __copy__(self):
        """A point owns a socket, so it cannot be copied."""
        raise NotImplementedError("Cannot do copies of CommPoint")

    def _printInfo(self, info: str):
        now = datetime.datetime.now()
        logger.debug("CommPoint[" + str(now) + "]: " + info)

    def setDebug(self, st: bool = True):
        """Log a timestamped line for each send, receive and peek (debug level)."""
        self._debug = st
        if st:
            enable_console_logging(logging.DEBUG)

    def _drop(self):
        """Close the connection after a failure that leaves the stream unusable."""
        try:
            self._sock.close()
        finally:
            self._begun = False

    def sendData(self, data: Dict) -> str:
        """
        Pickle ``data`` and send it as one frame.

        Returns:
            An empty string on success, otherwise the error text. A payload larger
            than ``max_frame_size`` is not sent. If the send times out (the peer
            stopped reading), the connection is closed.

        Raises:
            RuntimeError: If the point has not begun.
        """
        if not self._begun:
            raise RuntimeError("Cannot send data in not-begun commpoint")
        mydictser = pickle.dumps(data)
        if len(mydictser) > self._max_frame_size:
            return "payload of {} bytes exceeds the limit of {} bytes".format(
                len(mydictser), self._max_frame_size
            )
        frame = _HEADER.pack(_MAGIC, PROTOCOL_VERSION, len(mydictser)) + mydictser
        try:
            if self._debug:
                self._printInfo("Sending " + str(len(mydictser)) + " bytes...")
            # Without a timeout, sendall blocks for good once the peer stops
            # reading and the socket buffers fill up.
            self._sock.settimeout(
                self._send_timeout if self._send_timeout > 0 else None
            )
            self._sock.sendall(frame)
            self._sock.settimeout(None)
            if self._debug:
                self._printInfo("\tSent ok.")
            return ""
        except socket.timeout:
            self._drop()  # part of the frame may be on the wire
            return "send timed out after {} s; connection closed".format(
                self._send_timeout
            )
        except Exception as e:
            return str(e)

    def _recv_exactly(self, size: int) -> bytes:
        """Read exactly ``size`` bytes; raise if the peer closes first."""
        chunks = []
        pending = size
        while pending > 0:
            chunk = self._sock.recv(pending)
            if chunk == b"":
                raise (RuntimeError("Connection closed while receiving"))
            chunks.append(chunk)
            pending -= len(chunk)
        return b"".join(chunks)

    def readData(self, timeout: float = 2.0) -> Tuple[str, Dict]:
        """
        Receive one frame and unpickle it.

        Args:
            timeout: Seconds to wait on each socket read (not on the whole
                message). A value of 0 or less blocks until data arrives.

        Returns:
            ``(error, data)``. ``error`` is empty on success; otherwise it holds
            the error text (``timed out`` when the peer is silent) and ``data``
            is ``None``.

        Raises:
            RuntimeError: If the point has not begun.
        """
        if not self._begun:
            raise RuntimeError("Cannot read data in not-begun commpoint")
        if timeout <= 0.0:
            timeout = None
        # Past the timeout the other side is taken to be gone.
        self._sock.settimeout(timeout)
        try:
            if self._debug:
                self._printInfo("Receiving...")
            head = self._recv_exactly(_HEADER.size)
            magic, version, datasize = _HEADER.unpack(head)
            if magic != _MAGIC:
                raise _FrameError(
                    "the peer does not speak this protocol (bad header {!r})".format(
                        head[:2]
                    )
                )
            if version != PROTOCOL_VERSION:
                raise _FrameError(
                    "protocol version {} received, {} expected; use the same "
                    "release on both sides".format(version, PROTOCOL_VERSION)
                )
            if datasize > self._max_frame_size:
                raise _FrameError(
                    "frame of {} bytes exceeds the limit of {} bytes".format(
                        datasize, self._max_frame_size
                    )
                )
            data = self._recv_exactly(datasize)
            if self._allow_pickle:
                result = pickle.loads(data)
            else:
                result = _RestrictedUnpickler(io.BytesIO(data)).load()
            if self._debug:
                self._printInfo("\tReceived " + str(len(data)) + " bytes.")
            res = ""
        except Exception as e:
            result = None
            res = str(e)
            if isinstance(e, _FrameError):
                self._sock.settimeout(None)
                self._drop()  # the unread rest of the stream is meaningless
                return res, result
        self._sock.settimeout(None)  # do not leave the timeout set for other calls
        return res, result

    def checkDataToRead(self):
        """
        Return True if a read would not block: data is waiting, or the peer has
        closed the connection. Never blocks.
        """
        if not self._begun:
            raise RuntimeError("Cannot check data in not-begun commpoint")
        if self._debug:
            self._printInfo("Peeking...")
        ready_to_read, _, _ = select.select([self._sock], [], [], 0)
        if ready_to_read:
            return True
        return False


class ServerCommPoint(BaseCommPoint):
    """Point that listens on a port and accepts one client."""

    def __init__(
        self,
        po: int,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
        allow_pickle: bool = False,
        send_timeout: float = 10.0,
    ):
        """
        Bind to the port and start listening. If the port is busy, retry every
        13 seconds, up to 10 times, before giving up.

        On POSIX systems the socket sets ``SO_REUSEADDR``, so a server restarted
        right after a run does not have to wait for the old connection to leave
        TIME_WAIT. It does not allow two live servers on one port.

        The socket always binds to ``0.0.0.0`` (every interface). That works the
        same on a bare host and inside a container that publishes the port to
        another host (see ``examples/lunar_lander_container``).
        :meth:`BaseCommPoint.get_ip` only supplies the address shown in messages;
        it does not affect the bind.

        Args:
            po: TCP port, between 20000 and 49151.
            max_frame_size: See :class:`BaseCommPoint`.
            allow_pickle: See :class:`BaseCommPoint`.
            send_timeout: See :class:`BaseCommPoint`.

        Raises:
            OSError: If the port stays busy, or the bind fails for another reason.
        """
        self._servip = BaseCommPoint.get_ip()
        super().__init__(
            kind=BaseCommPoint.Kind.SERVER,
            port=po,
            ipv4="0.0.0.0",
            max_frame_size=max_frame_size,
            allow_pickle=allow_pickle,
            send_timeout=send_timeout,
        )
        finish = False
        tries = 0
        while not finish:
            try:
                self._basesock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if os.name == "posix":
                    # Not on Windows, where it would let a second server take the
                    # port of a live one.
                    self._basesock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._basesock.bind((self._ipv4, self._port))
                finish = True
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    tries += 1
                    if tries > 10:
                        logger.error("Too many tries. Aborting")
                        raise
                    logger.warning(
                        f"Port {self._servip}:{po} already in use. "
                        f"Retrying in 13 secs ({tries})..."
                    )
                    time.sleep(13)
                else:
                    logger.error(f"Socket error: {e}")
                    raise
        self._basesock.listen(1)
        logger.info("---> Server comm point listening")

    def __str__(self) -> str:
        return "Server listening at {}:{}, began: {}".format(
            self._servip, self._port, self._begun
        )

    def begin(self, timeoutaccept: float) -> str:
        """
        Wait for a client to connect. Any connection already open is closed first.

        Args:
            timeoutaccept: Seconds to wait; must be positive.

        Returns:
            An empty string once a client is connected, ``timeout`` if none
            arrived in time, otherwise the error text.
        """
        if timeoutaccept <= 0.0:
            raise ValueError("Timeoutaccept must be > 0.0")
        if self._begun:
            self.end()
        # Past the timeout the other side is taken to be gone.
        self._basesock.settimeout(timeoutaccept)
        try:
            self._sock, _ = self._basesock.accept()
            self._disable_nagle(self._sock)
            self._begun = True
            self._basesock.settimeout(None)
            return ""
        except socket.timeout:
            self._basesock.settimeout(None)
            return "timeout"
        except Exception as e:
            return str(e)

    def end(self) -> str:
        """
        Close the connection with the client. The listening socket stays open.

        Returns:
            An empty string on success (also if there was no connection),
            otherwise the error text.
        """
        if self._begun:
            try:
                self._sock.close()
                self._begun = False
                return ""
            except Exception as e:
                return str(e)
        return ""


class ClientCommPoint(BaseCommPoint):
    """Point that connects to a server."""

    def __init__(
        self,
        ip: str,
        po: int,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
        allow_pickle: bool = False,
        send_timeout: float = 10.0,
    ):
        """
        Args:
            ip: IPv4 address of the server.
            po: TCP port of the server, between 20000 and 49151.
            max_frame_size: See :class:`BaseCommPoint`.
            allow_pickle: See :class:`BaseCommPoint`.
            send_timeout: See :class:`BaseCommPoint`.
        """
        self._myip = BaseCommPoint.get_ip()
        super().__init__(
            kind=BaseCommPoint.Kind.CLIENT,
            ipv4=ip,
            port=po,
            max_frame_size=max_frame_size,
            allow_pickle=allow_pickle,
            send_timeout=send_timeout,
        )

    def __str__(self) -> str:
        return "Client at {} to connect to {}:{}, began: {}".format(
            self._myip, self._ipv4, self._port, self._begun
        )

    def begin(self) -> str:
        """
        Connect to the server. Any connection already open is closed first.

        Returns:
            An empty string on success, otherwise the error text.
        """
        if self._begun:
            self.end()
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._disable_nagle(self._sock)
            # connect() returns once the OS has queued the connection, even if the
            # server has not called accept() yet.
            self._sock.connect((self._ipv4, self._port))
            self._begun = True
            return ""
        except Exception as e:
            return str(e)

    def end(self) -> str:
        """
        Close the connection.

        Returns:
            An empty string on success (also if there was no connection),
            otherwise the error text.
        """
        if self._begun:
            try:
                self._sock.close()
                self._begun = False
                return ""
            except Exception as e:
                return str(e)
        return ""


if __name__ == "__main__":
    user_input = input("IP to connect to (empty if server): ")
    if len(user_input) == 0:
        comm = ServerCommPoint(49054)
        print("[{}] prepared to begin".format(str(comm)))
        err = comm.begin(60.0)
        if err:
            raise RuntimeError(
                "No one has connected to this server before timeout: " + err
            )
        print("[{}] connected".format(str(comm)))
        ind = 0
        while True:
            data = comm.readData(30.0)
            if data[0]:
                print("\t#{}. Received data {}".format(ind, data[1]))
            else:
                raise RuntimeError("\t#{}. Some error receiving data".format(ind))
            comm.sendData(data[1])
            print("\t\tSent response")
            ind += 1
    else:
        port = input("Port to connect to: ")
        comm = ClientCommPoint(ip=user_input, po=int(port))
        print("[{}] prepared to connect".format(str(comm)))
        comm.begin()
        print("[{}] connected".format(str(comm)))
        ind = 0
        while True:
            comm.sendData({"d": 54.54, "i": ind})
            print("\t\tSent data")
            data = comm.readData(30.0)
            if data[0]:
                print("\t#{}. Received response {}".format(ind, data[1]))
            else:
                raise RuntimeError("\t#{}. Some error receiving response".format(ind))
            time.sleep(10)
            ind += 1

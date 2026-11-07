# SPDX-License-Identifier: GPL-3.0-only

"""
Socket transport for rl_spin_decoupler.

A server point and a client point exchange pickled dictionaries over one TCP
connection. Each message is a frame: the payload length as 4 bytes (big endian)
followed by the pickled payload. Failures are reported by returning the error
text (an empty string means success) rather than by raising, except for misuse
such as sending through a point that has not begun.

(c) Juan-Antonio Fernández-Madrigal, 2025
https://babel.isa.uma.es/jafma
"""

import datetime
import ipaddress
import pickle
import select
import socket
import struct
import time
from enum import Enum
from typing import Dict, Tuple


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
        datachunkmaxsize: int = 4096,
        port: int = 49054,
        ipv4: str = "127.0.0.1",
    ):
        """
        Args:
            kind: Role of the point.
            datachunkmaxsize: Positive integer. It is validated and stored, but
                the transport does not use it yet.
            port: TCP port, between 20000 and 49151.
            ipv4: IPv4 address the point refers to: the address to bind for a
                server, the address to connect to for a client.

        Raises:
            TypeError: If ``kind`` is not a :class:`Kind`.
            ValueError: If any other argument is invalid.
        """
        if not isinstance(kind, BaseCommPoint.Kind):
            raise (TypeError("Expected a Kind argument, got {}".format(type(kind))))
        if (not isinstance(datachunkmaxsize, int)) or not (0 < datachunkmaxsize):
            raise (
                ValueError(
                    "The max. size of data chunks {} is invalid".format(
                        datachunkmaxsize
                    )
                )
            )
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
        self._datachunkmaxsize = datachunkmaxsize
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
        print("CommPoint[" + str(now) + "]: " + info, flush=True)

    def setDebug(self, st: bool = True):
        """Print a timestamped line for each send, receive and peek."""
        self._debug = st

    def sendData(self, data: Dict) -> str:
        """
        Pickle ``data`` and send it as one frame.

        Returns:
            An empty string on success, otherwise the error text.

        Raises:
            RuntimeError: If the point has not begun.
        """
        if not self._begun:
            raise RuntimeError("Cannot send data in not-begun commpoint")
        mydictser = pickle.dumps(data)
        frame = struct.pack("!I", len(mydictser)) + mydictser
        try:
            if self._debug:
                self._printInfo("Sending " + str(len(mydictser)) + " bytes...")
            self._sock.sendall(frame)
            if self._debug:
                self._printInfo("\tSent ok.")
            return ""
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
            head = self._recv_exactly(4)
            datasize = struct.unpack("!I", head)[0]
            data = self._recv_exactly(datasize)
            result = pickle.loads(data)
            if self._debug:
                self._printInfo("\tReceived " + str(len(data)) + " bytes.")
            res = ""
        except Exception as e:
            result = None
            res = str(e)
        self._sock.settimeout(None)  # do not leave the timeout set for other calls
        return res, result

    def checkDataToRead(self):
        """
        Return True if a read would not block: data is waiting, or the peer has
        closed the connection. Never blocks.
        """
        if not self._begun:
            raise RuntimeError("Cannot send data in not-begun commpoint")
        if self._debug:
            self._printInfo("Peeking...")
        ready_to_read, _, _ = select.select([self._sock], [], [], 0)
        if ready_to_read:
            return True
        return False


class ServerCommPoint(BaseCommPoint):
    """Point that listens on a port and accepts one client."""

    def __init__(self, po: int):
        """
        Bind to the port and start listening. If the port is busy, retry every
        13 seconds, up to 10 times, before giving up.

        The socket always binds to ``0.0.0.0`` (every interface). That works the
        same on a bare host and inside a container that publishes the port to
        another host (see ``examples/lunar_lander_container``).
        :meth:`BaseCommPoint.get_ip` only supplies the address shown in messages;
        it does not affect the bind.

        Args:
            po: TCP port, between 20000 and 49151.

        Raises:
            OSError: If the port stays busy, or the bind fails for another reason.
        """
        self._servip = BaseCommPoint.get_ip()
        super().__init__(kind=BaseCommPoint.Kind.SERVER, port=po, ipv4="0.0.0.0")
        finish = False
        tries = 0
        while not finish:
            try:
                self._basesock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._basesock.bind((self._ipv4, self._port))
                finish = True
            except OSError as e:
                if e.errno == socket.errno.EADDRINUSE:
                    tries += 1
                    if tries > 10:
                        print("Too many tries. Aborting")
                        raise
                    print(
                        f"Port {self._servip}:{po} already in use. "
                        f"Retrying in 13 secs ({tries})..."
                    )
                    time.sleep(13)
                else:
                    print(f"Socket error: {e}")
                    raise
        self._basesock.listen(1)
        print("---> Server comm point listening")

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

    def __init__(self, ip: str, po: int):
        """
        Args:
            ip: IPv4 address of the server.
            po: TCP port of the server, between 20000 and 49151.
        """
        self._myip = BaseCommPoint.get_ip()
        super().__init__(kind=BaseCommPoint.Kind.CLIENT, ipv4=ip, port=po)

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

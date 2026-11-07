# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Use the private vulnerability reporting of the
GitHub repository (Security tab), or write to the maintainers listed in
`pyproject.toml`. We will acknowledge the report and agree on a way to disclose it.

## What the library protects against, and what it does not

The two endpoints exchange pickled Python objects over TCP, and unpickling can run
arbitrary code. By default the receiver only accepts plain data and NumPy arrays and
scalars (`allow_pickle=False`), bounds the size of every frame (`max_frame_size`) and
closes the connection on a malformed header. That reduces the exposure; it does not make
pickle a safe format for untrusted peers.

- Run the two endpoints on a trusted network, or through an SSH tunnel (see the
  `*_remote` examples). Do not expose the RL port to the Internet.
- There is no authentication or encryption. Anyone who can reach the port can talk to
  the RL side.
- Use `allow_pickle=True` only between endpoints you trust.

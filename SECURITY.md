# Security Policy

> **Modified-file notice:** this policy derives from the Apache-2.0 crforge security policy and was
> updated by NextoCR.

## Scope

NextoCR is an offline game simulator for research and training purposes. It is not production infrastructure,
but we still take security issues seriously.

The ZMQ bridge binds to `127.0.0.1` by default. `--allow-remote` is an explicit development escape
hatch that exposes an unauthenticated control protocol on every interface; do not use it on an
untrusted network.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **GitHub Private Advisory** (preferred): Go to
   the [Security Advisories](https://github.com/theohugo/NextoCR/security/advisories) tab and create
   a new private advisory.
2. **Email**: Send details to the repository maintainer via their GitHub profile.

Please do **not** open a public issue for security vulnerabilities.

## Response

We will acknowledge reports within 7 days and aim to release a fix promptly. We appreciate your help
keeping NextoCR safe.

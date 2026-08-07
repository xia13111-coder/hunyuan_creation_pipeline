# Security Policy

## Reporting a vulnerability

Do not report exposed credentials, private data, command injection, arbitrary
file access, unsafe archive extraction, or remote-code-execution issues in a
public issue.

Use GitHub private vulnerability reporting when available. Otherwise contact
the repository owner through a private channel listed on their GitHub profile.
Include:

- affected version or commit;
- minimal reproduction steps;
- impact and required privileges;
- whether credentials or user data may have been exposed;
- a suggested mitigation, if known.

Do not include live secrets or private production assets in the report. Replace
them with synthetic values and revoke exposed credentials immediately.

## Supported versions

Security fixes are applied to the default branch and the most recent published
release. Older snapshots and generated Docker bundles are not guaranteed to
receive backports.

## Security boundaries

This project launches Blender, Isaac Sim, model runtimes, and optional cloud
clients as subprocesses. Treat STEP/STP, GLB, USD, images, MDL files, model
checkpoints, archives, and configuration received from untrusted sources as
potentially hostile. Run untrusted inputs in an isolated account or container,
with minimal filesystem access and no unrelated credentials.

The repository must not contain real `.env` files, API keys, cloud secrets,
model-access tokens, private reference photographs, or generated run outputs.
Use `.env.example` only as a list of variable names.

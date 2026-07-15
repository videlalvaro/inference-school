# Security Policy

## Supported versions

Security fixes are applied to the current `main` branch. This project does not
currently maintain older release branches.

## Reporting a vulnerability

Do not disclose a suspected vulnerability in a public issue, discussion, or
pull request.

When this repository is public and private vulnerability reporting is enabled,
use the **Report a vulnerability** button on the repository's Security page:

https://github.com/videlalvaro/inference-school/security

If that button is unavailable, contact the repository owner through the GitHub
profile at https://github.com/videlalvaro and request a private reporting
channel. Include the affected version, reproduction steps, impact, and any
proposed mitigation. Do not include secrets or personal data in the initial
contact.

You should receive an acknowledgment within seven days. Confirmed issues will
be handled through a private GitHub security advisory until a fix and disclosure
timeline are agreed.

## Execution boundary

Inference School intentionally compiles and runs learner-authored Swift and Metal code.
The packaged Studio uses App Sandbox and a user-selected workspace. Its host
signature includes the client entitlement required to run its bundled WebKit
diagram and editor views. Built-in checks do not upload learner code, but the
packaged runner currently inherits the Studio sandbox, including that
entitlement. Command-line checks are not sandboxed by Inference School and run with the
permissions of the terminal process that launches them. Only run code and
course content you trust.
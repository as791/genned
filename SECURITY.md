# Security policy

## Supported versions

Only the [latest release](https://github.com/as791/genned/releases/latest) gets fixes.
Please update before reporting.

## Reporting a vulnerability

Please **don't open a public issue** for a security problem. Report it privately:

1. Go to https://github.com/as791/genned/security/advisories/new
   (Security tab → **Report a vulnerability**).
2. Describe the problem, how to reproduce it, the app version and your Android version.

Only the maintainers can see the report. You'll get a reply as soon as possible, and we'll
agree on a fix and on when to disclose it. With your permission, you'll be credited in the
release notes.

## What counts

Examples of what we want to hear about:

- Anything that lets data leave the phone: the app is meant to have no network access at
  all.
- Ways for another app to read Genned's history, files or screen captures.
- A crafted image or video that crashes the app or runs code (the decoders, metadata and
  Content Credentials parsers).
- Ways to make a Content Credentials check show "Verified" for a record that isn't
  validly signed by a trusted signer.
- Problems with how release builds are signed or published.

## What doesn't

Detection mistakes are expected: an AI image the app misses, or a real photo it flags.
Genned gives an estimate, not proof, and it is known not to be robust to deliberate
adversarial attacks. Please report these with the **Wrong result** issue template instead.
